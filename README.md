# sars_cov_2_epitopes

Find epitopes in proteins to accelerate vaccine design

### Motivation

B-cells in our immune system reacts to specific parts of the antigens, called epitopes. Thus, epitopes can be used to develope vaccines, such that the immune system can develope antibodies against the virus.

In times of covid-19 spreading across the globe, the search for a vaccine is as urgent as never before. To accelerate this process, it would be very beneficial to identify the epitopes in the Sars-Cov-2 proteins.

### Project

In previous work called BepiPred by Jespersen et. al. [1], a random forest is trained to detect epitopes in protein sequences. Here, the goal is to improve on this model, with a two-step approach:

The pipeline consists of two models:

* A Random Forest model that takes amino-acids and their context as input and predicts for each single amino acid the probability that it belongs to an epitope
* A CNN that takes short sequences (epitope candidates) as input and predicts whether or not it is a epitope (binary output)

Then, Sars-Cov-2 proteins are processed in the following way:

* feed sequences through the Random Forest model to get the epitope-probability per amino acid
* find  areas of high probability in the sequence --> these are possible epitope candidates
* take these candidates and input them into the predictive model --> this yields the probability that a candidate is actually an epitope

The output would be the epitope candidate of each protein which got the highest rating from the predictive model

The code for this pipeline is in the notebook [pipeline_epitopes](https://github.com/NinaWie/sars_cov_2_epitopes/blob/master/pipeline_epitopes.ipynb)

### Dependencies

To install dependencies with a conda environment, run

```
conda env create -f environment.yml
```


### Bibligraphy

Jespersen, Martin Closter, et al. "BepiPred-2.0: improving sequence-based B-cell epitope prediction using conformational epitopes." Nucleic acids research 45.W1 (2017): W24-W29.