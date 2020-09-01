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

Generated files:
- `Cache/preprocessed_comments.json` and `Cache/preprocessed_test_need_kg.json` with preprocessed data.
- `Cache/predicted_kg_comment_top10_tfidf.json` with the top 10 comments prediction result for each question.

#### Stage 2: BERT


### Knowledge-based answer prediction



 

