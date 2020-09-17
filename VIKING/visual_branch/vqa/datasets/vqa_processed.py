"""
Preprocess a train/test pair of interim json data files.
Caption: Use NLTK or split function to get tokens. 
"""
import json
import pickle
from typing import List
import click
from nltk import pos_tag
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
import yaml
from tqdm import tqdm
import os


def extract_concepts(tokens: List[List[str]]) -> List[List[str]]:
    concepts = []
    for t in tqdm(tokens, desc="pos tagging"):
        pos_tags = pos_tag(t)
        norm_w = []
        for w in pos_tags:  # count the question words
            if w[1].startswith("VB"):
                normalized_word = nltk.stem.WordNetLemmatizer().lemmatize(
                    w[0], "v"
                )
                if normalized_word == "be" or normalized_word == "do":
                    continue
            elif w[1].startswith("NN"):
                normalized_word = nltk.stem.WordNetLemmatizer().lemmatize(
                    w[0], "n"
                )
            elif w[1].startswith("JJ"):
                normalized_word = nltk.stem.WordNetLemmatizer().lemmatize(
                    w[0], "a"
                )
            else:
                continue

            norm_w.append(normalized_word)
        concepts.append(norm_w)
    return concepts


def get_top_answers(examples: List, top_n=3000) -> List[str]:
    c = Counter([x["answer"] for x in examples])
    top_ans = c.most_common(top_n)
    return [x[0] for x in top_ans]


def remove_OoV_examples(examples: List, ans_vocab: List[str]) -> List:
    return [ex for ex in examples if ex["answer"] in ans_vocab]


def get_vocab(docs: List[List[str]], minwcount: int = 0) -> List[str]:
    # count up the number of words
    counter: Counter = Counter()
    for d in docs:
        counter.update(d)

    # remove rare words
    remove_w = [w for w, n in counter.items() if n <= minwcount]
    for w in remove_w:
        counter.pop(w)

    return list(counter.keys())


def preprocess(train_file: str, out_dir: str, params: dict):
    trainset = json.load(open(train_file))
    trainset = [x for x in trainset if not x["need_external_knowledge"]]

    top_answers = get_top_answers(trainset, top_n=params["nans"])
    aid_to_ans = [a for i, a in enumerate(top_answers)]
    ans_to_aid = {a: i for i, a in enumerate(top_answers)}

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(f"{out_dir}/aid_to_ans.pkl", "wb") as f:
        pickle.dump(aid_to_ans, f)

    with open(f"{out_dir}/ans_to_aid.pkl", "wb") as f:
        pickle.dump(ans_to_aid, f)

    # Remove examples if answer is not in top answers
    trainset = remove_OoV_examples(trainset, top_answers)

    questions = []
    for ex in trainset:
        question = word_tokenize(str(ex["question"]).lower())
        questions.append(question)

    concepts = extract_concepts(questions)
    cid_to_concept = get_vocab(concepts, params["concept_mincount"])
    concept_to_cid = {c: i for i, c in enumerate(cid_to_concept)}

    with open(f"{out_dir}/cid_to_concept.pkl", "wb") as f:
        pickle.dump(cid_to_concept, f)

    with open(f"{out_dir}/concept_to_cid.pkl", "wb") as f:
        pickle.dump(concept_to_cid, f)

    q_vocab = get_vocab(questions, params["minwcount"])
    q_vocab = ["EOS"] + q_vocab + ["START"] + ["UNK"]
    wid_to_word = {i: w for i, w in enumerate(q_vocab)}
    word_to_wid = {w: i for i, w in enumerate(q_vocab)}

    with open(f"{out_dir}/wid_to_word.pkl", "wb") as f:
        pickle.dump(wid_to_word, f)

    with open(f"{out_dir}/word_to_wid.pkl", "wb") as f:
        pickle.dump(word_to_wid, f)


@click.command()
@click.argument("path_opt", type=click.Path(exists=True))
def main(path_opt):
    with open(path_opt, "r") as f:
        params = yaml.load(f)

    preprocess(
        params["dataset"]["train"],
        params["dataset"]["dict_dir"],
        params["dataset"],
    )


if __name__ == "__main__":
    main()