#! /usr/bin/env python3
# coding=utf-8
"""
DeLorean decoding for abductive reasoning.
"""

import argparse
from operator import add
from typing import List
import json

import numpy as np
from tqdm import trange

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from transformers import GPT2Tokenizer
from transformers.modeling_gpt2 import GPT2LMHeadModel


SMALL_CONST = 1e-15
BIG_CONST = 1e10


def to_var(x, requires_grad=False, volatile=False, device="cuda"):
    if torch.cuda.is_available() and device == "cuda":
        x = x.cuda()
    elif device != "cuda":
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def read_inputs(input_file):
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


def top_k_filter(logits, k, probs=False, device='cuda'):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.

    Args:
        probs (bool): Whether `logits` is indeed probabilities
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)


def get_input_embeds(embedding, logits , o1_onehot=None, o2_onehot=None, device='cuda'):
    """
    embedding.shape = [50257, 1024]
    """
    probs = F.softmax(logits, dim=-1)
    if o1_onehot is not None:
        probs = torch.cat(
            (o1_onehot.type(torch.FloatTensor), probs.type(torch.FloatTensor)),
            dim=1)
    if o2_onehot is not None:
        probs = torch.cat(
            (probs.type(torch.FloatTensor), o2_onehot.type(torch.FloatTensor)),
            dim=1)
    probs = probs.to(device)
    return torch.matmul(probs, embedding.weight)


def get_token_from_logits(logits, temperature=1.0, top_k=1):
    """
    logits.shape = [batch_size]
    """
    # normalize
    logits = top_k_filter(logits, k=top_k)
    probs = F.softmax(logits, dim=-1)

    # greedy
    _, last = torch.topk(probs, k=1, dim=-1)

    return last


def get_text_from_logits(logits, tokenizer, temperature=1.0, top_k=1):
    output_so_far = None
    for i in range(logits.shape[1]):
        last = get_token_from_logits(logits[:,i,:], temperature, top_k)

        # update context/output_so_far appending the new token
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)

    text = tokenizer.decode(output_so_far.tolist()[0])
    text = text.replace('\n', ' ')

    return text


def generate_abductive_explanation(
    model=None,
    tokenizer=None,
    device='cuda',
    o1_text="",
    o2_text="",
    length=10,
    max_length=20,
    stepsize=0.02,
    mix_rate=0.5,
    temperature_first=1.0,
    temperature_backward=1.0,
    temperature_forward=1.0,
    top_k=1,
    num_passes=3,
    num_backward_iters=1,
    seed=0,
    no_cuda=False,
    verbose=False
):
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Figure out o1 o2 text
    tokenized_o1_text = tokenizer.encode(tokenizer.bos_token + o1_text)
    tokenized_o2_text = tokenizer.encode(o2_text + tokenizer.eos_token)

    if verbose:
        print("= o1 | o2 =")
        print(tokenizer.decode(tokenized_o1_text))
        print(tokenizer.decode(tokenized_o2_text))
        print()

    # Generate with DeLorean decoding
    _, candidate_list = delorean_decoding(
        model=model,
        tokenizer=tokenizer,
        o1=tokenized_o1_text,
        o2=tokenized_o2_text,
        device=device,
        length=length,
        max_length=max_length,
        stepsize=stepsize,
        mix_rate=mix_rate,
        temperature_first=temperature_first,
        temperature_backward=temperature_backward,
        temperature_forward=temperature_forward,
        top_k=top_k,
        num_passes=num_passes,
        num_backward_iters=num_backward_iters,
        verbose=verbose
    )
    if device == "cuda":
        torch.cuda.empty_cache()

    return candidate_list


def delorean_decoding(
    model,
    tokenizer,
    o1=None,
    o2=None,
    device="cuda",
    length=10,
    max_length=20,
    stepsize=0.02,
    mix_rate=0.5,
    temperature_first=1.0,
    temperature_backward=1.0,
    temperature_forward=1.0,
    top_k=1,
    num_passes=3,
    num_backward_iters=1,
    verbose=False
):
    """
    Perform DeLorean decoding for abductive reasoning.

    Largely the same code as in counterfactual reasoning (counterfactual_main.py),
    except for the loss (in the backward_pass) and a few details.
    """

    # Prepare one-hot representations for O1 and O2
    o1_t = torch.tensor(o1, device=device, dtype=torch.long)
    while len(o1_t.shape) < 2:
        o1_t = o1_t.unsqueeze(0)
    output_so_far = o1_t

    o1_onehot = torch.LongTensor(o1_t.shape[0], o1_t.shape[1], tokenizer.vocab_size)
    o1_onehot = o1_onehot.to(device)
    o1_onehot.zero_()
    o1_onehot.scatter_(2, o1_t.unsqueeze(-1), 1)
    # use a very small temperature to mimic one-hot after softmax
    o1_logits = o1_onehot.type(torch.FloatTensor) / 0.00001

    o2_t = torch.tensor(o2, device=device, dtype=torch.long)
    while len(o2_t.shape) < 2:
        o2_t = o2_t.unsqueeze(0)
    o2_onehot = torch.LongTensor(o2_t.shape[0], o2_t.shape[1], tokenizer.vocab_size)
    o2_onehot = o2_onehot.to(device)
    o2_onehot.zero_()
    o2_onehot.scatter_(2, o2_t.unsqueeze(-1), 1)


    ## The initialization pass to initialize the generation (its logits)

    past = None
    last_embeds = None
    logits_so_far = None
    for i in range(length):
        # run model forward to obtain unperturbed logits
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            last_embeds = model.get_input_embeddings()(last)

            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :-1])
                o1_past = past

        unpert_logits, past, unpert_all_hidden = model(past=past, inputs_embeds=last_embeds)
        unpert_logits = unpert_logits[:, -1, :] / temperature_first

        unpert_logits = unpert_logits.unsqueeze(1)
        logits_so_far = unpert_logits if logits_so_far is None else torch.cat((logits_so_far, unpert_logits), dim=1)

        last_embeds = get_input_embeds(model.get_input_embeddings(), unpert_logits / 0.01, device=device)

    if verbose:
        print("[First pass]: ", get_text_from_logits(logits_so_far, tokenizer, temperature=1.0, top_k=top_k))

    unpert_logits_h = logits_so_far


    ## Iteratively perturb the generation through Forward and Backward passes

    pert_logits = unpert_logits_h

    grad_norms = None
    candidate_list = []
    for t in trange(num_passes, ascii=True):

        if verbose:
            print()
            print("=" * 20)
            print('Pass ', t)
            print("=" * 20)

        pert_logits = backward_pass(
            pert_logits,
            model,
            tokenizer,
            o2_onehot=o2_onehot,
            o2=o2_t,
            stepsize=stepsize,
            temperature=temperature_backward,
            top_k=top_k,
            num_backward_iters=num_backward_iters,
            device=device,
            verbose=verbose
        )

        pert_logits, forward_text = forward_pass(
            pert_logits,
            model,
            tokenizer,
            o1_logits=o1_logits,
            o1=o1_t,
            length=length,
            max_length=max_length,
            mix_rate=mix_rate,
            temperature=temperature_forward,
            top_k=top_k,
            device=device,
            verbose=verbose
        )

        candidate_list.append(forward_text)

    return output_so_far, candidate_list


def forward_pass(
    logits,
    model,
    tokenizer,
    o1_logits=None,
    o1=None,
    length=10,
    max_length=20,
    mix_rate=0.5,
    temperature=1.0,
    top_k=1,
    device="cuda",
    verbose=False
):
    """
    Args:
        length: length of the hypothesis whose logits are updated through the
            forward-backward passes. I.e., `N` in the paper
        max_length: we allow the forward pass to generate more than N tokens if those are
            needed to obtain complete sentences. See section 3.1 (last paragraph) in the
            paper. Extra tokens will be truncated.
    """
    assert logits.shape[1] == length
    h_logits = logits

    past = None
    last_embeds = None
    logits_so_far = None
    logits_so_far_complete = None
    for i in range(max_length):
        # Run model forward to obtain unperturbed logits
        if past is None:
            o1_embeds = get_input_embeds(model.get_input_embeddings(), o1_logits, device=device)
            last_embeds = o1_embeds[:, -1:, :]

            if o1_logits.shape[1] > 1:
                _, past, _ = model(inputs_embeds=o1_embeds[:, :-1, :])

        elif i == 0:
            last_embeds = model.get_input_embeddings()(o1[:, -1:])

        unpert_logits, past, unpert_all_hidden = model(past=past, inputs_embeds=last_embeds)
        unpert_logits = unpert_logits[:, -1, :] / temperature

        if i < length:
            # Mix backward logits and forward logits, Eq.(3) in the paper
            pert_logits = mix_rate * unpert_logits + (1-mix_rate) * h_logits[:,i,:]
        else:
            # Continue to complete the text
            pert_logits = unpert_logits

        pert_logits = pert_logits.unsqueeze(1)
        if i < length:
            logits_so_far = pert_logits if logits_so_far is None else torch.cat((logits_so_far, pert_logits), dim=1)
        logits_so_far_complete = pert_logits if logits_so_far_complete is None else torch.cat((logits_so_far_complete, pert_logits), dim=1)

        # Use a small temperature (0.1) so that the soft token representation is sharper,
        # and closer to a one-hot representation
        last_embeds = get_input_embeds(model.get_input_embeddings(), pert_logits / 0.1, device=device)

    # Sample a text
    forward_text = get_text_from_logits(logits_so_far_complete, tokenizer, temperature=1.0, top_k=top_k)
    if verbose:
        print("[Forward]: ", forward_text)

    return logits_so_far, forward_text


def backward_pass(
    logits,
    model,
    tokenizer,
    o2_onehot=None,
    o2=None,
    stepsize=0.01,
    temperature=1.0,
    top_k=1,
    num_backward_iters=3,
    device="cuda",
    verbose=False
):

    # Set logits to a list just for ease of programming and experimentation
    logits = [logits]

    # Accumuated gradients w.r.t the logits
    grad_accumulator = [(np.zeros(p.shape).astype("float32")) for p in logits]

    # Accumulate perturbations for num_backward_iters
    for i in range(num_backward_iters):
        if verbose:
            print("\n-------Iteration------- ", i + 1)

        # Compute the perturbed logits
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device) for p_ in grad_accumulator
        ]
        perturbed_logits = list(map(add, logits, [curr_perturbation[0]]))

        # Compute the norms of the logits for normalizing the gradients later
        perturbed_logits_norms_all = [
            torch.norm(p_) for index, p_ in enumerate(perturbed_logits)
        ]

        inputs_embeds = get_input_embeds(model.get_input_embeddings(),
                                         perturbed_logits[0] / temperature,
                                         o2_onehot=o2_onehot,
                                         device=device)

        # Compute loss
        o2_length = o2_onehot.shape[1]
        all_logits, _, _ = model(inputs_embeds=inputs_embeds)
        assert all_logits.shape[1] == perturbed_logits[0].shape[1] + o2_length
        o2_logits = all_logits[:, -o2_length-1:-1, :]  # exclude the last step (which is a prediction)
        assert o2_logits.shape[1] == o2_length

        loss = torch.nn.CrossEntropyLoss()(
            o2_logits.view(-1, o2_logits.size(-1)),
            o2.view(-1))
        if verbose:
            print("loss: %.4f" % (loss.data.cpu().numpy()))

        # Compute gradients
        loss.backward()

        # Compute gradient norms
        grad_norms_logits_all = [
            (torch.norm(p_.grad) + SMALL_CONST) for index, p_ in enumerate([curr_perturbation[0]])
        ]
        # Normalize and scale the gradients
        grad = [
            -stepsize * (p_.grad / grad_norms_logits_all[index] * perturbed_logits_norms_all[index]).data.cpu().numpy()
            for index, p_ in enumerate([curr_perturbation[0]])
        ]

        # Accumulate gradients
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # Reset gradients
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # Remove logits from the graph
        new_logits = []
        for p_ in logits:
            new_logits.append(p_.detach())
        logits = new_logits

        if verbose:  # inspect the temporary text after the backward pass
            _grad_accumulator = [to_var(torch.from_numpy(p_), requires_grad=True, device=device) for p_ in [grad_accumulator[0]]]
            _pert_logits = list(map(add, logits, _grad_accumulator))
            text = get_text_from_logits(_pert_logits[0], tokenizer, temperature=1.0, top_k=top_k)
            print("[Backward]: ", text)

    # Apply the accumulated gradients to the logits
    grad_accumulator = [to_var(torch.from_numpy(p_), requires_grad=True, device=device) for p_ in grad_accumulator]
    pert_logits = list(map(add, logits, grad_accumulator))

    return pert_logits[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model", type=str, default="gpt2-medium",
        help="pretrained model name or path to local checkpoint")
    parser.add_argument(
        "--length", type=int, default=10,
        help="Length of generated text. Not used in the counterfactual setting because the generation length "
             "is set to the length of the original story ending.")
    parser.add_argument(
        "--max_length", type=int, default=20,
        help="Max length of generated text. We allow the forward pass to generate more than `length` tokens if "
             "those are needed to obtain complete sentences. See section 3.1 (last paragraph) for details.")
    parser.add_argument("--mix_rate", type=float, default=0.5, help="Weight of mixing backward and forward logits in the forward pass.")
    parser.add_argument("--temperature_first", type=float, default=1.0, help="Temperature of logits used in the initialization pass.")
    parser.add_argument("--temperature_backward", type=float, default=1.0, help="Temperature of logits used in the backward pass.")
    parser.add_argument("--temperature_forward", type=float, default=1.0, help="Temperature of logits used in the forward pass.")
    parser.add_argument("--top_k", type=int, default=1, help="Top-k sampling from logits.")
    parser.add_argument("--stepsize", type=float, default=0.02, help="learning rate in the backward pass.")
    parser.add_argument("--num_backward_iters", type=int, default=1, help="Number of backpropagation iterations in a Backward pass.")
    parser.add_argument("--num_passes", type=int, default=3, help="Number of passes to interleave Forward and Backward.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbose", action="store_true", help="Print intermediate states to help with tuning / debugging.")
    parser.add_argument("--input_file", type=str, default="", help="Input data in json format.")
    parser.add_argument("--output_dir", type=str, default="", help="Output dir.")

    args = parser.parse_args()

    # Set the device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # Load pretrained model
    model = GPT2LMHeadModel.from_pretrained(args.pretrained_model, output_hidden_states=True)
    model.to(device)
    model.eval()
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model)
    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    candidate_output = './{}/abductive_output_np{}_nbi{}.json'.format(
        args.output_dir, args.num_passes, args.num_backward_iters)

    records = read_inputs(args.input_file)

    processed = set()
    candidate_list = []

    with open(candidate_output, 'w') as fw:
        for r in records:
            # We use O2<E>O1 as the left context. Yet prepending O2 here is
            # minor to the overall performance. See the latest paper
            # (https://arxiv.org/abs/2010.05906, section 4.1) for more details.
            o1_text = '<|endoftext|>'.join([r['obs2'], r['obs1']])
            o2_text = r['obs2']

            # The original dataset can include repeated instances.
            # We keep track and skip instances that are already processed
            if o1_text not in processed:
                vars(args)['o1_text'], vars(args)['o2_text'] = o1_text, o2_text

                candidate_list = generate_abductive_explanation(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    o1_text=o1_text,
                    o2_text=o2_text,
                    length=args.length,
                    max_length=args.max_length,
                    stepsize=args.stepsize,
                    mix_rate=args.mix_rate,
                    temperature_first=args.temperature_first,
                    temperature_forward=args.temperature_forward,
                    temperature_backward=args.temperature_backward,
                    top_k=args.top_k,
                    num_passes=args.num_passes,
                    num_backward_iters=args.num_backward_iters,
                    seed=args.seed,
                    no_cuda=args.no_cuda,
                    verbose=args.verbose)

                processed.add(o1_text)

            d = {
                'O1': o1_text,
                'O2': o2_text,
                'H_Candidates': candidate_list
            }
            fw.write(json.dumps(d) + '\n')
            fw.flush()
