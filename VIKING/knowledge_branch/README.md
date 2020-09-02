## Knowlegde QA Branch

The `knowledge QA branch` predicts an answer for a sample classified as external knowledge is needed.

1. First, the model retrieves the relevant external knowledge from SemArt dataset comments to answer the question.

2. Then, the model uses the retrieved comment to generate an answer to the question.

Code by [Zihua Liu](https://github.com/Zihua-Liu) and [Chentao Ye](https://github.com/chentaoye). Maintenance by [Noa Garcia](https://github.com/noagarcia).


### Two-stage external knowledge retrieval

#### Dependencies 

```bash
conda install -c anaconda nltk
conda install -c anaconda scipy
conda install -c anaconda numpy
conda install -c anaconda pandas
conda install -c conda-forge transformers=2.1.1 
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

#### Data
- The `comments` folder contains `semart_train.csv`, `semart_val.csv`, and `semart_test.csv`, 
which are obtained from the original SemArt dataset and are used as the knowledge base in our model.

#### Stage 1: TF-IDF
For a given question, this module predicts the top-10 most relevant comments in the knowledge base. 

```bash
# Preprocessing questions and comments
python preprocess.py

# Run TF-IDF scores between questions and comments
python tf_idf.py
```

*Generated files*:
- `Cache/preprocessed_comments.json` and `Cache/preprocessed_test_need_kg.json` with preprocessed data.
- `Cache/predicted_kg_comment_top10_tfidf.json` with the top 10 comments prediction result for each question.

#### Stage 2: BERT

In the second stage, a BERT model is used to re-rank the 10 comments from the first stage. 

```bash
# Obtain data
unzip Data/precomputed_top10.zip -d Data/

# Process the outputs from TF-IDF model
python toBert.py

# Finetune BERT model
python trainBert.py

# Re-rank comments
python toTestBert.py
python testBert.py
```

*Notes*: 
1. The top 10 comments for the train, validation, and test have already been pre-computed and stored in `Data/` directory.
2. Instead of `trainBert.py`, you can download the [fine-tuned model](https://drive.google.com/open?id=14_9iA5f6eBSrnTwE2rhQRI3J3KNta8dl) and store the pretrained weights and the config file in `Models/Bert/` folder.

*Output*:
- Results will be stored in `Cache/xlnet-pipeline.json`.


### Knowledge-based answer prediction

This module is based on XLNet.

Train the model:

```
python toXLNet.py
bash trainXLNet.sh
```

Predict answers for the test set:
```
bash testXLNet.sh
```

*Notes*: 
1. Instead of `trainXLNet.py`, you can download the [pretrained model](https://drive.google.com/open?id=14_9iA5f6eBSrnTwE2rhQRI3J3KNta8dl) and store the pretrained weights and the config file in `Models/XLNet/` folder.

*Output*:
- The predict answers for the knowledge QA branch are be stored in `/Models/XLNet/predictions_.json`


### Accuracy

The reported accuracy of the `Knowledge QA branch` alone is of 47.6%. That is 1,726 exact matches over 3,626 samples classified 
as *external knowledge requiered* by the `Modality Selector`.

The total accuracy of the whole system is 55.5%, with 1,726 exact matches in the `Knowledge QA branch` 
and 1,000 exact matches in the `Visual QA branch` over a total of 4,912 samples in the test set.

