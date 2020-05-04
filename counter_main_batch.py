#! /usr/bin/env python3
# coding=utf-8

# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
"""

import argparse
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange

from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel


SMALL_CONST = 1e-15
BIG_CONST = 1e10

def to_var(x, requires_grad=False, volatile=False, device="cuda"):
    if torch.cuda.is_available() and device == "cuda":
        x = x.cuda()
    elif device != "cuda":
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]

def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
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
    logits = top_k_filter(logits, k=top_k)  # + SMALL_CONST
    probs = F.softmax(logits, dim=-1)

    # sample or greedy
    if False: #sample:
        last = torch.multinomial(probs, num_samples=1)
    else:
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
    #print(text)

    return text


def run_story_example(
    pretrained_model="gpt2-small",
    o1_text="",
    o2_text="",
    length=20,
    stepsize=0.02,
    mix_rate=0.5,
    temperature_first=1.0,
    temperature_backward=1.0,
    temperature_forward=1.0,
    temperature_o1=0.01,
    top_k=1,
    sample=False,
    num_passes=3,
    num_iterations=1,
    perturb_o1=False,
    gamma=1.5,
    seed=0,
    no_cuda=False,
    repetition_penalty=1.0,
    anli_jsonl_file=""
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(pretrained_model, output_hidden_states=True)
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out o1 o2 text
    tokenized_o1_text = tokenizer.encode(tokenizer.bos_token + o1_text)
    tokenized_o2_text = tokenizer.encode(o2_text + tokenizer.eos_token)

    print("= o1 | o2 =")
    print(tokenizer.decode(tokenized_o1_text))
    print(tokenizer.decode(tokenized_o2_text))
    print()

    # generate unperturbed and perturbed texts

    #unpert_gen_tok_text, _, _ = generate_text(
    _, forward_list = generate_text(
        model=model,
        tokenizer=tokenizer,
        o1=tokenized_o1_text,
        o2=tokenized_o2_text,
        device=device,
        perturb=True,
        length=length,
        stepsize=stepsize,
        mix_rate=mix_rate,
        temperature_first=temperature_first,
        temperature_backward=temperature_backward,
        temperature_forward=temperature_forward,
        temperature_o1=temperature_o1,
        top_k=top_k,
        sample=sample,
        num_passes=num_passes,
        num_iterations=num_iterations,
        perturb_o1=perturb_o1,
        gamma=gamma,
        repetition_penalty=repetition_penalty
    )
    if device == "cuda":
        torch.cuda.empty_cache()

    return forward_list


#def generate_text(
#    model,
#    tokenizer,
#    o1=None,
#    o2=None,
#    device="cuda",
#    perturb=True,
#    length=100,
#    stepsize=0.02,
#    temperature=1.0,
#    top_k=1,
#    sample=False,
#    num_passes=3,
#    num_iterations=1,
#    gamma=1.5,
#    repetition_penalty=1.0
#):
#
#    output_so_far = None
#    if o1:
#        o1_t = torch.tensor(o1, device=device, dtype=torch.long)
#        while len(o1_t.shape) < 2:
#            o1_t = o1_t.unsqueeze(0)
#        output_so_far = o1_t
#
#        o1_onehot = torch.LongTensor(o1_t.shape[0], o1_t.shape[1], tokenizer.vocab_size)
#        o1_onehot = o1_onehot.to(device)
#        o1_onehot.zero_()
#        o1_onehot.scatter_(2, o1_t.unsqueeze(-1), 1)
#
#    if o2:
#        o2_t = torch.tensor(o2, device=device, dtype=torch.long)
#        while len(o2_t.shape) < 2:
#            o2_t = o2_t.unsqueeze(0)
#        o2_onehot = torch.LongTensor(o2_t.shape[0], o2_t.shape[1], tokenizer.vocab_size)
#        o2_onehot = o2_onehot.to(device)
#        o2_onehot.zero_()
#        o2_onehot.scatter_(2, o2_t.unsqueeze(-1), 1)
#
#    ## The 1st left-to-right pass
#
#    past = None
#    last = None
#    for i in range(length):
#
#        # Get past/probs for current output, except for last word
#        # Note that GPT takes 2 inputs: past + current_token
#
#        # run model forward to obtain unperturbed
#        if past is None and output_so_far is not None:
#            last = output_so_far[:, -1:]
#            if output_so_far.shape[1] > 1:
#                _, past, _ = model(output_so_far[:, :-1])
#
#        unpert_logits, past, unpert_all_hidden = model(last, past=past)
#        unpert_logits = unpert_logits[:, -1, :] / temperature  # + SMALL_CONST
#
#        # + repetition penalty
#        for token_idx in set(output_so_far[0].tolist()):
#            if unpert_logits[0, token_idx] < 0:
#                unpert_logits[0, token_idx] *= repetition_penalty
#            else:
#                unpert_logits[0, token_idx] /= repetition_penalty
#
#        # normalize
#        unpert_logits = top_k_filter(unpert_logits, k=top_k)  # + SMALL_CONST
#        unpert_probs = F.softmax(unpert_logits, dim=-1)
#
#        # sample or greedy
#        if sample:
#            last = torch.multinomial(unpert_probs, num_samples=1)
#        else:
#            _, last = torch.topk(unpert_probs, k=1, dim=-1)
#
#        # update context/output_so_far appending the new token
#        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
#
#        print(tokenizer.decode(output_so_far.tolist()[0]))
#
#    unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
#    unpert_logits_h = unpert_logits[:, -length-1:-1, :] # logits of tokens in h
#
#    #print("o1 length: ", o1_t.shape[1])
#    #print("logits length: ", unpert_logits.shape[1])
#    #print("logits_h length: ", unpert_logits_h.shape[1])
#    #print("length: ", length)
#    #print("length output_so_far: ", output_so_far.shape[1])
#
#
#    ## Iteratively perturb the generation
#
#    logits_h = unpert_logits_h
#    grad_norms = None
#    for t in trange(num_passes, ascii=True):
#
#        ## Right-to-left perturbation
#        perturb_right_to_left(
#            logits_h,
#            model,
#            tokenizer,
#            o1_onehot=o1_onehot,
#            o2_onehot=o2_onehot,
#            o2=o2_t,
#            grad_norms=grad_norms,
#            stepsize=stepsize,
#            temperature=temperature,
#            top_k=top_k,
#            num_iterations=num_iterations,
#            gamma=gamma,
#            device=device
#        )
#
#        ## Left-to-right perturbation
#
#
#    return output_so_far

def generate_text(
    model,
    tokenizer,
    o1=None,
    o2=None,
    device="cuda",
    perturb=True,
    length=100,
    stepsize=0.02,
    mix_rate=0.5,
    temperature_first=1.0,
    temperature_backward=1.0,
    temperature_forward=1.0,
    temperature_o1=0.01,
    top_k=1,
    sample=False,
    num_passes=3,
    num_iterations=1,
    perturb_o1=False,
    gamma=1.5,
    repetition_penalty=1.0
):

    output_so_far = None
    if o1:
        o1_t = torch.tensor(o1, device=device, dtype=torch.long)
        while len(o1_t.shape) < 2:
            o1_t = o1_t.unsqueeze(0)
        output_so_far = o1_t

        o1_onehot = torch.LongTensor(o1_t.shape[0], o1_t.shape[1], tokenizer.vocab_size)
        o1_onehot = o1_onehot.to(device)
        o1_onehot.zero_()
        o1_onehot.scatter_(2, o1_t.unsqueeze(-1), 1)

        if perturb_o1:
            temp = temperature_o1
        else:
            temp = 0.00001 # very small to mimic one-hot after softmax
        o1_logits = o1_onehot.type(torch.FloatTensor) / temp

    if o2:
        o2_t = torch.tensor(o2, device=device, dtype=torch.long)
        while len(o2_t.shape) < 2:
            o2_t = o2_t.unsqueeze(0)
        o2_onehot = torch.LongTensor(o2_t.shape[0], o2_t.shape[1], tokenizer.vocab_size)
        o2_onehot = o2_onehot.to(device)
        o2_onehot.zero_()
        o2_onehot.scatter_(2, o2_t.unsqueeze(-1), 1)

    ## The 1st left-to-right pass

    # Get past/probs for current output, except for last word
    # Note that GPT takes 2 inputs: past + current_token

    # run model forward to obtain unperturbed
    unpert_logits, _, _ = model(torch.cat([o1_t, o2_t], dim=-1))
    o2_length = o2_t.shape[1]
    o2_logits = unpert_logits[:, -o2_length-1:-1, :] # TODO(h): exclude the last step (which is a prediction)
    assert unpert_logits.shape[1] == o1_t.shape[1] + o2_length
    assert o2_logits.shape[1] == o2_length

    # O2 loss
    loss = torch.nn.CrossEntropyLoss()(o2_logits.view(-1, o2_logits.size(-1)), o2_t.view(-1))
    print("[First pass] o2 loss: ", loss.data.cpu().numpy())

    #print("[First pass]: ", get_text_from_logits(logits_so_far, tokenizer, temperature=1.0, top_k=top_k))


    ## Iteratively perturb the generation

    if perturb_o1:
        raise NotImplementedError
        #pert_logits = torch.cat((o1_logits.to(device), unpert_logits_h), dim=1)
    else:
        pert_logits = o2_logits
    grad_norms = None
    forward_list = []
    for t in trange(num_passes, ascii=True):

        print("=" * 20)
        print('Pass ', t)
        print("=" * 20)

        if t == 0 and False: #TODO
            pass
        else:
            ## Right-to-left perturbation
            pert_logits, grad_norms, _ = perturb_right_to_left(
                pert_logits,
                #past,
                model,
                tokenizer,
                o1_onehot=o1_onehot,
                o2_onehot=o2_onehot,
                o2=o2_t,
                grad_norms=grad_norms,
                stepsize=stepsize,
                temperature=temperature_backward,
                top_k=top_k,
                num_iterations=num_iterations,
                gamma=gamma,
                device=device
            )

        if t < num_passes - 1:
            fwd_length = o2_length
        else:
            fwd_length = o2_length * 2

        ## Left-to-right perturbation
        pert_logits, forward_text = perturb_left_to_right(
            pert_logits,
            model,
            tokenizer,
            o1_logits=o1_logits,
            o1_onehot=o1_onehot,
            o2_onehot=o2_onehot,
            #length=length,
            length=fwd_length,
            mix_rate=mix_rate,
            temperature=temperature_forward,
            top_k=top_k,
            perturb_o1=perturb_o1,
            device=device
        )

        forward_list.append(forward_text)

    return output_so_far, forward_list


def perturb_left_to_right(
    logits,
    model,
    tokenizer,
    o1_logits=None,
    o1_onehot=None,
    o2_onehot=None,
    length=100,
    mix_rate=0.5,
    temperature=1.0,
    top_k=1,
    perturb_o1=False,
    device="cuda"
):

    if perturb_o1:
        raise NotImplementedError
        #o1_length = o1_logits.shape[1]
        #assert logits.shape[1] == o1_length + length
        #o1_logits = logits[:, :o1_length, :] # perturbed o1 logits
        #h_logits = logits[:, o1_length:, :]
    else:
        assert logits.shape[1] == length or logits.shape[1] == length / 2
        if logits.shape[1] == length:
            h_logits = logits
        else:
            h_logits = torch.cat((logits, torch.zeros_like(logits, device=device)), dim=1)

    past = None
    last_embeds = None
    logits_so_far = None
    for i in range(length):

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None:
            o1_embeds = get_input_embeds(model.get_input_embeddings(), o1_logits, device=device)
            last_embeds = o1_embeds[:, -1, :].unsqueeze(1)

            if o1_logits.shape[1] > 1:
                _, past, _ = model(inputs_embeds=o1_embeds[:, :-1, :])

        unpert_logits, past, unpert_all_hidden = model(past=past, inputs_embeds=last_embeds)
        unpert_logits = unpert_logits[:, -1, :] / temperature  # + SMALL_CONST

        pert_logits = mix_rate * unpert_logits + (1-mix_rate) * h_logits[:,i,:]

        pert_logits = pert_logits.unsqueeze(1)
        logits_so_far = pert_logits if logits_so_far is None else torch.cat((logits_so_far, pert_logits), dim=1)

        last_embeds = get_input_embeds(model.get_input_embeddings(), pert_logits / 0.1, device=device) #TODO

    if perturb_o1:
        raise NotImplementedError
        #logits_so_far = torch.cat((o1_logits, logits_so_far), dim=1)

    #output_so_far = o1

    #past = None
    #last_embeds = None
    #logits_so_far = None
    #for i in range(length):

    #    # Get past/probs for current output, except for last word
    #    # Note that GPT takes 2 inputs: past + current_token

    #    # run model forward to obtain unperturbed
    #    if past is None and output_so_far is not None:
    #        last = output_so_far[:, -1:]
    #        last_embeds = model.get_input_embeddings()(last)

    #        if output_so_far.shape[1] > 1:
    #            _, past, _ = model(output_so_far[:, :-1])

    #    unpert_logits, past, unpert_all_hidden = model(past=past, inputs_embeds=last_embeds)
    #    unpert_logits = unpert_logits[:, -1, :] / temperature  # + SMALL_CONST

    #    pert_logits = mix_rate * unpert_logits + (1-mix_rate) * logits[:,i,:]

    #    pert_logits = pert_logits.unsqueeze(1)
    #    logits_so_far = pert_logits if logits_so_far is None else torch.cat((logits_so_far, pert_logits), dim=1)

    #    last_embeds = get_input_embeds(model.get_input_embeddings(), pert_logits, device=device)

    forward_text = get_text_from_logits(logits_so_far, tokenizer, temperature=1.0, top_k=top_k)
    print("[Forward]: ", forward_text)

    return logits_so_far, forward_text


def perturb_right_to_left(
    logits, # logits of h
    #past,
    model,
    tokenizer,
    o1_onehot=None,
    o2_onehot=None,
    o2=None,
    grad_norms=None,
    stepsize=0.01,
    temperature=1.0,
    top_k=1,
    num_iterations=3,
    gamma=1.5,
    device="cuda"
):

    # TODO(h): Set logits to a list to preserve the PPLM code structure
    logits = [logits]

    # Generate inital perturbed logits
    grad_accumulator = [(np.zeros(p.shape).astype("float32")) for p in logits]
    #grad_accumulator_past = [(np.zeros(p.shape).astype("float32")) for p in past]

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    for i in range(num_iterations):
        print("\n-------Iteration------- ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device) for p_ in grad_accumulator
        ]
        #curr_perturbation_past = [
        #    to_var(torch.from_numpy(p_), requires_grad=True, device=device) for p_ in grad_accumulator_past
        #]

        # Compute using perturbed logits
        perturbed_logits = list(map(add, logits, curr_perturbation))
        #perturbed_past = list(map(add, past, curr_perturbation_past))

        perturbed_logits_norms = [
            torch.norm(p_, dim=-1) for index, p_ in enumerate(perturbed_logits)
        ]

        topk_lg, topk_index = torch.topk(perturbed_logits[0][0,-5,:], 5)
        print(topk_lg.data.cpu().numpy())
        print(topk_index.data.cpu().numpy())
        print(perturbed_logits_norms[0][0,-5].data.cpu().numpy())
        print('~'*20)

        #inputs_embeds = get_input_embeds(model.get_input_embeddings(),
        #                                 perturbed_logits[0] / temperature, # temperature
        #                                 o1_onehot=o1_onehot,
        #                                 #o2_onehot=o2_onehot,
        #                                 device=device)
        #all_logits, _, _ = model(inputs_embeds=inputs_embeds)
        #o2_length = o2_onehot.shape[1]
        #o2_logits = all_logits[:, -o2_length-1:-1, :] # TODO(h): exclude the last step (which is a prediction)
        #assert all_logits.shape[1] == o1_onehot.shape[1] + o2_length
        #assert o2_logits.shape[1] == o2_length

        ## O2 loss
        #loss_1 = torch.nn.CrossEntropyLoss()(o2_logits.view(-1, o2_logits.size(-1)), o2.view(-1))
        loss_1 = 0

        loss_2 = torch.nn.CrossEntropyLoss()(perturbed_logits[0].view(-1, perturbed_logits[0].size(-1)), o2.view(-1))
        loss = 0.1 * loss_1 + loss_2
        print("o2 loss:", loss.data.cpu().numpy())

        loss_per_iter.append(loss.data.cpu().numpy())

        # compute gradients
        loss.backward()

        # calculate gradient norms
        if grad_norms is not None and False: # TODO
            print('grad_norms 1')
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            print('grad_norms 2')
            grad_norms = [
                (torch.norm(p_.grad, dim=-1) + SMALL_CONST) for index, p_ in enumerate(curr_perturbation)
            ]

        # print(grad_norms[0][0,-5].data.cpu().numpy())
        #
        # topk_grad, topk_index = torch.topk(torch.abs(curr_perturbation[0].grad[0,-5,:]), 5)
        # print(topk_grad.data.cpu().numpy())
        # print(topk_index.data.cpu().numpy())

        ## normalize gradients
        #grad = [
        #    -stepsize * (p_.grad / grad_norms[index] ** gamma).data.cpu().numpy()
        #    for index, p_ in enumerate(curr_perturbation)
        #]

        grad = [
            -stepsize * (p_.grad / grad_norms[index].unsqueeze(-1) * perturbed_logits_norms[index].unsqueeze(-1)).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        topk_grad, topk_index = torch.topk(torch.abs(torch.FloatTensor(grad[0])[0,-5,:]), 5)
        print(topk_grad.data.cpu().numpy())
        print(topk_index.data.cpu().numpy())
        grad_temp_norms = [
            torch.norm(torch.FloatTensor(p_), dim=-1) for index, p_ in enumerate(grad)
        ]
        print(grad_temp_norms[0][0,-5])

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_logits = []
        for p_ in logits:
            new_logits.append(p_.detach())
        logits = new_logits


        _grad_accumulator = [to_var(torch.from_numpy(p_), requires_grad=True, device=device) for p_ in grad_accumulator]
        _pert_logits = list(map(add, logits, _grad_accumulator))
        text = get_text_from_logits(_pert_logits[0], tokenizer, temperature=1.0, top_k=top_k)
        print("[Backward]: ", text)


    # apply the accumulated perturbations to the past
    grad_accumulator = [to_var(torch.from_numpy(p_), requires_grad=True, device=device) for p_ in grad_accumulator]
    pert_logits = list(map(add, logits, grad_accumulator))

    return pert_logits[0], grad_norms, loss_per_iter



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2", #gpt2-medium
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument("--o1_text", type=str, default="Stephen was at a party.", help="o1")
    parser.add_argument("--o2_text", type=str, default="He checked it but it was completely broken.", help="o2")
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--mix_rate", type=float, default=0.5)
    parser.add_argument("--temperature_first", type=float, default=1.0)
    parser.add_argument("--temperature_backward", type=float, default=1.0)
    parser.add_argument("--temperature_forward", type=float, default=1.0)
    parser.add_argument("--temperature_o1", type=float, default=0.01)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--num_passes", type=int, default=3)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--perturb_o1", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="Penalize repetition. More than 1.0 -> less repetition",
    )
    parser.add_argument("--anli_jsonl_file", type=str, default="", help="gold")

    args = parser.parse_args()

    forward_output = './output_counter1/counter_newlen_output_topk_{}_sts_{}_mr_{}_np_{}_ni_{}_tfst_{}_tbd_{}_tfd_{}.json'.format(
        args.top_k,
        args.stepsize, args.mix_rate, args.num_passes, args.num_iterations, args.temperature_first,
        args.temperature_backward, args.temperature_forward)

    records = read_jsonl_lines(args.anli_jsonl_file)

    rm_dup = set()

    with open(forward_output, 'w') as fw, open(forward_output+'.txt', 'w') as fw_txt:
        for r in records:
            o1_text = ' '.join([r['premise'], r['counterfactual']])
            o2_text = r['original_ending']
            if o1_text in rm_dup:
                continue
            else:
                rm_dup.add(o1_text)
            vars(args)['o1_text'], vars(args)['o2_text'] = o1_text, o2_text
            forward_list = run_story_example(**vars(args))
            # forward_list = [x[len(o1_text) + len('<|endoftext|>') + 1:] for x in forward_list]
            d = {
                'premise': r['premise'],
                'initial': r['initial'],
                'counterfactual': r['counterfactual'],
                'original_ending': o2_text,
                'H_Candidates': forward_list
            }
            # print(d)
            fw.write(json.dumps(d) + '\n')
            fw_txt.write(forward_list[-1] + '\n')

