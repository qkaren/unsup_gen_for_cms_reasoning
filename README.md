# unsupervised_story

The code is adapted from the [Plug-and-Play code](https://github.com/huggingface/transformers/tree/master/examples/pplm) developed in [this paper](https://arxiv.org/abs/1912.02164).


## Counterfactual reasoning

Run
```
sh run_test_counter.sh
```
See [code here](https://github.com/qkaren/unsupervised_story/blob/master/counter_main.py#L705) for details of all arguments.

Toggle `--verbose` to control how much you want the program to output intermediate results.

**Example output:**

(without setting `--verbose`)
```
= o1 | o2 =
<|endoftext|>Ned wanted to learn to play basketball. He hung out in the library studying English.
Soon he felt confident enough to shoot a few hoops himself. The team watched him play and they cheered. Ned's skills improved as he practiced.<|endoftext|>
====================
Pass  0
====================
[Forward]:   He was a confident enough kid to a few basketball games. The team watched him play and they cheered. Ned's skills improved as he practiced. He
====================
Pass  1
====================
[Forward]:   He was a good enough student that a few of his friends were team captains.  "I was a good student," he said. "I
...
```
where each line starting with `[Forward]` is the resulting generation at each pass.


## Abductive reasoning

Run
```
sh run_test.sh
```
See [code here](https://github.com/qkaren/unsupervised_story/blob/master/main.py#L737) for details of all arguments.

Toggle `--verbose` to control how much you want the program to output intermediate results.

**Example output:**

(without setting `--verbose`)
```
= o1 | o2 =
<|endoftext|>It always wastes my time.<|endoftext|>I work at a place that believes in teamwork.
It always wastes my time.<|endoftext|>

[First pass]:   We. other.. their.........
====================
Pass  0
====================
[Forward]:  <|endoftext|>It always wastes my time.<|endoftext|>I work at a place that believes in teamwork. We can all, and have, all the more to the people that are
====================
Pass  1
====================
[Forward]:  <|endoftext|>It always wastes my time.<|endoftext|>I work at a place that believes in teamwork. The only other team that is as all the more so more like that is     
```
where each line starting with `[Forward]` is the resulting generation at each pass, in a format: 
```
[Forward]: <|endoftext|>[O2]<|endoftext|>[O1] Generation
```

