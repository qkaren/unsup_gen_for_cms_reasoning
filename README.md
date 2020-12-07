# Back to the Future: DeLorean Decoding for Commonsense Reasoning

This repo hosts the code for the following paper:

[Back to the Future: Unsupervised Backprop-based Decoding for Counterfactual and Abductive Commonsense Reasoning](https://arxiv.org/abs/2010.05906)                         
*Lianhui Qin,  Vered Shwartz, Peter West, Chandra Bhagavatula, Jena Hwang, Ronan Le Bras, Antoine Bosselut, Yejin Choi   
EMNLP 2020*

- The code consists of the implementations for the two tasks, namely *counterfactual reasoning* and *abductive reasoning*, respectively. 
- Example small data is included in `data/`
- Ranking code is included in `ranking/`
- Decoding and ranking results will be put in `output/`

## Counterfactual Reasoning

### Decoding
Run the following cmd to do DeLorean decoding for counterfactual reasoning, on the example data in `data/counterfactual/small_data.json`
```
sh run_counterfactual_main.sh
```
Results are written to `output/counterfactual/`, which include the generated hypotheses using different hyperparameters (*\#forward-backward passes* and *\#backward iterations*). These results are then to be ranked in the following.

### Ranking
Run the following cmds to rank the hypotheses
```
cd ranking/
sh run_counterfactual_ranking.sh
```
Ranked results are written to `output/counterfactual/ranking`


## Abducive Reasoning

The code and usage are largely the same as those of counterfactual reasoning. We write different code files for different data processing, loss functions, etc.

### Decoding
Run the following cmd to do DeLorean decoding for abductive reasoning, on the example data in `data/abductive/small_data.json`
```
sh run_abductive_main.sh
```
Results are written to `output/abductive/`, which include the generated hypotheses using different hyperparameters (*\#forward-backward passes* and *\#backward iterations*). These results are then to be ranked in the following.

### Ranking
Run the following cmds to rank the hypotheses
```
cd ranking/
sh run_abductive_ranking.sh
```
Ranked results are written to `output/abductive/ranking`

      
<br/>          

*Acknowledgement: the decoding and ranking code uses [Huggingface Transformers](https://github.com/huggingface). The decoding code is adapted (though with large changes) from the [Plug-and-Play LM code](https://github.com/huggingface/transformers/tree/master/examples/pplm).*
