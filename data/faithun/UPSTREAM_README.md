# FaithUn: Toward Faithful Forgetting in Language Models by Investigating the Interconnectedness of Knowledge

The FaithUn dataset serves as a benchmark for evaluating the unlearning of real-world entities.

The paper is available at the following link: [LINK](https://arxiv.org/abs/2502.19207).

This repository provides the FaithUn dataset along with detailed information on dataset splits.
  
  
## The dataset file (FaithUn_Dataset.json)

"FaithUnBench" is a new benchmark first introduced in our paper "FaithUn: Toward Faithful Forgetting in Language Models by Investigating the Interconnectedness of Knowledge."

This dataset is designed to investigate and evaluate the faithfulness of unlearning methods.

We constructed the benchmark from Wikidata triples (www.wikidata.org) covering 200 well-known individuals.

Each entity (e.g., Wikidata ID Q76 refers to Barack Obama) contains four types of question sets:

1. Base QA datasets  
2. Paraphrased QA datasets  
3. Multi-hop QA datasets  
4. Same-answer QA datasets  

In the FaithUnBench.json file, each entity includes the following categories:

1. hop1_unlearn → corresponds to Base QA datasets, used during unlearning. Each question forms a cluster for evaluation.  
2. hop1_test → corresponds to Paraphrased QA datasets, used for evaluation.  
3. hop2_test → corresponds to Multi-hop QA datasets, used for evaluation.  
4. same_object → corresponds to Same-answer QA datasets, used for evaluation.  

Additionally, each question includes both answers and false answer options to support evaluation.  

  
## Splits (forget set, retaining set, test set)

We concatenated all clusters of entities sequentially, resulting in 664 clusters in total.

These clusters were then split into the forget set (5%), retaining set (10%), and test set (70%). The indices for each split are provided in the files:

1. forget_set_split.txt  
2. retaining_set_split.txt  
3. test_set_split.txt  
