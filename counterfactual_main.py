#! /usr/bin/env python3
# coding=utf-8
"""
DeLorean decoding for counterfactual reasoning.
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


def top_k_filter(logits, k, probs=False):
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


def get_input_embeds(embedding, logits, o1_onehot=None, o2_onehot=None, device='cuda'):
    """
    Gets embedding of `logits` (through soft mixing).

    Args:
        o1_onehot: If provided, prepend o1 embedding to logits embedding
        o2_onehot: If provided, append o2 embedding to logits embedding
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


def generate_counterfactual_story_endings(
    model=None,
    tokenizer=None,
    device='cuda',
    o1_text="",
    o2_text="",
    max_length=20,
    stepsize=0.02,
    mix_rate=0.5,
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
    tokenized_o2_text = tokenizer.encode(o2_text)[1:]  # delete the "." token we appended before

    if verbose:
        print("= o1 | o2 =")
        print(tokenizer.decode(tokenized_o1_text))
        print(tokenizer.decode(tokenized_o2_text))
        print()

    # Generate counterfactual story endings with DeLorean decoding
    _, candidate_list = delorean_decoding(
        model=model,
        tokenizer=tokenizer,
        o1=tokenized_o1_text,
        o2=tokenized_o2_text,
        device=device,
        max_length=max_length,
        mix_rate=mix_rate,
        temperature_forward=temperature_forward,
        top_k=top_k,
        stepsize=stepsize,
        num_backward_iters=num_backward_iters,
        num_passes=num_passes,
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
    mix_rate=0.5,
    temperature_forward=1.0,
    top_k=1,
    stepsize=0.02,
    num_backward_iters=1,
    num_passes=3,
    verbose=False
):
    """
    Perform DeLorean decoding for abductive reasoning.

    Largely the same code as in abductive reasoning (abductive_main.py),
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


    ## The initialization pass to initialize the generation (its logits)

    # Run model forward to obtain unperturbed logits
    unpert_logits, _, _ = model(torch.cat([o1_t, o2_t], dim=-1))
    o2_length = o2_t.shape[1]
    o2_logits = unpert_logits[:, -o2_length-1:-1, :]  # exclude the last step which is a prediction
    assert unpert_logits.shape[1] == o1_t.shape[1] + o2_length
    assert o2_logits.shape[1] == o2_length

    if verbose:
        # O2 loss
        loss = torch.nn.CrossEntropyLoss()(o2_logits.view(-1, o2_logits.size(-1)), o2_t.view(-1))
        print("[First pass] recon loss: ", loss.data.cpu().numpy())


    ## Iteratively perturb the generation through Forward and Backward passes

    pert_logits = o2_logits

    candidate_list = []
    for t in trange(num_passes, ascii=True):

        if verbose:
            print()
            print("=" * 20)
            print('Pass ', t)
            print("=" * 20)

        if t > 0:
            pert_logits = backward_pass(
                pert_logits,
                model,
                tokenizer,
                o2=o2_t,
                stepsize=stepsize,
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
            length=o2_length,
            max_length=o2_length + 20,
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
            last_embeds = o1_embeds[:, -1, :].unsqueeze(1)

            if o1_logits.shape[1] > 1:
                _, past, _ = model(inputs_embeds=o1_embeds[:, :-1, :])

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

    # Sample a text, and only extract the first sentence
    forward_text = get_text_from_logits(logits_so_far_complete, tokenizer, temperature=1.0, top_k=top_k)
    forward_text, _ = _extract_a_sentence(forward_text)
    if verbose:
        print("[Forward]: ", forward_text)

    return logits_so_far, forward_text


def backward_pass(
    logits,
    model,
    tokenizer,
    o2=None,
    stepsize=0.01,
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
        perturbed_logits = list(map(add, logits, curr_perturbation))

        # Compute the norms of the logits for normalizing the gradients later
        perturbed_logits_norms_all = [
            torch.norm(p_) for index, p_ in enumerate(perturbed_logits)
        ]

        # Compute loss
        loss = torch.nn.CrossEntropyLoss()(
            perturbed_logits[0].view(-1, perturbed_logits[0].size(-1)),
            o2.view(-1))
        if verbose:
            print("loss: %.4f" % (loss.data.cpu().numpy()))

        # Compute gradients
        loss.backward()

        # Compute gradient norms
        grad_norms_all = [
            (torch.norm(p_.grad) + SMALL_CONST) for index, p_ in enumerate(curr_perturbation)
        ]
        # Normalize and scale the gradients
        grad = [
            -stepsize * (p_.grad / grad_norms_all[index] * perturbed_logits_norms_all[index]).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
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
            _grad_accumulator = [to_var(torch.from_numpy(p_), requires_grad=True, device=device) for p_ in grad_accumulator]
            _pert_logits = list(map(add, logits, _grad_accumulator))
            text = get_text_from_logits(_pert_logits[0], tokenizer, temperature=1.0, top_k=top_k)
            print("[Backward]: ", text)

    # Apply the accumulated gradients to the logits
    grad_accumulator = [to_var(torch.from_numpy(p_), requires_grad=True, device=device) for p_ in grad_accumulator]
    pert_logits = list(map(add, logits, grad_accumulator))

    return pert_logits[0]


def _extract_a_sentence(text):
    """
    Extracts the first sentence in `text`.
    Returns the sentence and the remaining text.
    """
    # (1)
    sent_terminators = ['. ', '! ', '? ']
    min_tm_index = BIG_CONST
    for tm in sent_terminators:
        tm_index = text.find(tm)
        if tm_index == -1:
            tm_index = BIG_CONST
        min_tm_index = min(min_tm_index, tm_index)

    if min_tm_index < BIG_CONST:
        return text[:min_tm_index+1], text[min_tm_index+2:]

    # (2)
    sent_terminators = ['." ', '!" ', '?" ']
    for tm in sent_terminators:
        tm_index = text.find(tm)
        if tm_index == -1:
            tm_index = BIG_CONST
        min_tm_index = min(min_tm_index, tm_index)

    if min_tm_index < BIG_CONST:
        return text[:min_tm_index+2], text[min_tm_index+3:]

    return text, ""


def extract_three_sentences(text):
    """
    `text` is assumed to consist of three sentences. This function
    extracts and returns the three sentences.
    """
    s1, s23 = _extract_a_sentence(text)
    s2, s3 = _extract_a_sentence(s23)
    return s1, s2, s3


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

    candidate_output = './{}/counterfactual_output_np{}_nbi{}.json'.format(
        args.output_dir, args.num_passes, args.num_backward_iters)

    records = read_inputs(args.input_file)

    procssed = set()

    # `fw` outputs all results, `fw_text` outputs the cleaned results
    with open(candidate_output, 'w') as fw, open(candidate_output+'.txt', 'w') as fw_txt:
        for r in records:
            o1_text = ' '.join([r['premise'], r['counterfactual']])
            o2_text = r['original_ending']

            # The original dataset can include repeated instances.
            # We keep track and skip instances that are already processed
            if o1_text in procssed:
                continue
            else:
                procssed.add(o1_text)

            # o2_text has three sentences. We use DoLorean to generate one
            # sentence at a time. See Appendix A.2 in the paper for more details.
            o2_text_sents = extract_three_sentences(o2_text)

            o2_text_so_far = ""
            o1_text_so_far = ""
            o1_addon = o1_text

            for o2_sent in o2_text_sents:
                o1_text_so_far = o1_text_so_far.strip() + " " + o1_addon.strip()
                o2_text_so_far = o2_sent.strip()

                # We want to ensure a space token between o1_text and o2_text
                # during the decoding. To do so, here we append ". " so that
                # the GPT2 tokenizer later will not strip the space token. After
                # tokenization, we delete the "." token.
                o2_text_so_far = ". " + o2_text_so_far

                candidate_list = generate_counterfactual_story_endings(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    o1_text=o1_text_so_far,
                    o2_text=o2_text_so_far,
                    max_length=args.max_length,
                    stepsize=args.stepsize,
                    mix_rate=args.mix_rate,
                    temperature_forward=args.temperature_forward,
                    top_k=args.top_k,
                    num_passes=args.num_passes,
                    num_backward_iters=args.num_backward_iters,
                    seed=args.seed,
                    no_cuda=args.no_cuda,
                    verbose=args.verbose)

                d = {
                    'premise': r['premise'],
                    'initial': r['initial'],
                    'counterfactual': r['counterfactual'],
                    'original_ending': o2_text,
                    'counterfactual_so_far': o1_text_so_far,
                    'original_ending_so_far': o2_text_so_far,
                    'H_Candidates': candidate_list
                }
                fw.write(json.dumps(d) + '\n')
                fw_txt.write(candidate_list[-1] + '\n')

                o1_addon = candidate_list[-1]  # pick the last candidate
