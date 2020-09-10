"""
Preprocess a train/test pair of interim json data files.
Caption: Use NLTK or split function to get tokens. 
"""
import argparse
import json
import pdb
import pickle
from typing import List
import click
from nltk import pos_tag
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
import yaml
from tqdm import tqdm

# import pprint


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


# def buildup_concept_vocab(examples: List, mincount: int = 30):
#     counter: Counter = Counter()
#     for ex in examples:
#         for concepts in ex["concepts"]:
#             counter.update(concepts)

#     for c, n in counter:
#         if n <= mincount:
#             counter.pop(c)

#     concepts = list(counter.keys())
#     concept_to_cid = {c: i for i, c in enumerate(concepts)}

#     return concepts, concept_to_cid


# def encode_concepts(
#     examples, selected_concetps: List[str], concept_to_cid: Dict[str, int]
# ) -> List:
#     for ex in examples:
#         ex["concepts"] = [
#             concept
#             for concept in ex["concepts"]
#             if concept in selected_concetps
#         ]
#         ex["concepts_cid"] = [
#             concept_to_cid[concept] for concept in ex["concepts"]
#         ]
#     return examples


def get_top_answers(examples: List, top_n=3000) -> List[str]:
    c = Counter([x["answer"] for x in examples])
    top_ans = c.most_common(top_n)
    return [x[0] for x in top_ans]


def remove_OoV_examples(examples: List, ans_vocab: List[str]) -> List:
    return [ex for ex in examples if ex["answer"] in ans_vocab]


# def preprocess_questions(examples):
#     # print('Example of generated tokens after preprocessing some questions:')
#     for ex in examples:
#         s = ex["question"]
#         ex["question_words"] = word_tokenize(str(s).lower())
#     return examples


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


# def remove_long_tail_train(
#     examples, minwcount=0
# ) -> Tuple[List[Any], List[str]]:
#     # Replace words which are in the long tail (counted less than 'minwcount' times) by the UNK token.
#     # Also create vocab, a list of the final words.

#     questions = [ex["question_words"] for ex in examples]
#     vocab = get_vocab(questions, minwcount)

#     for ex in examples:
#         words = ex["question_words"]
#         question = [w if w in vocab else "UNK" for w in words]
#         ex["question_words_UNK"] = question

#     vocab.append("UNK")
#     return examples, vocab


# def remove_long_tail_test(examples, word_to_wid):
#     for ex in examples:
#         ex["question_words_UNK"] = [
#             w if w in word_to_wid else "UNK" for w in ex["question_words"]
#         ]
#     return examples


# def encode_question(examples, word_to_wid, maxlength=15, pad="left"):
#     # Add to tuple question_wids and question_length
#     for i, ex in enumerate(examples):
#         ex["question_length"] = min(
#             maxlength, len(ex["question_words_UNK"])
#         )  # record the length of this sequence
#         ex["question_wids"] = [0] * maxlength
#         question = ex["question_words_UNK"]
#         for k, w in enumerate(question):
#             # added by Yikang: To replace question mark to EOS
#             q = w.replace("?", "EOS")
#             if k < maxlength:
#                 if pad == "right":
#                     ex["question_wids"][k] = word_to_wid[q]
#                 else:  # ['pad'] == 'left'
#                     new_k = k + maxlength - len(ex["question_words_UNK"])
#                     ex["question_wids"][new_k] = word_to_wid[q]
#                 ex["seq_length"] = len(ex["question_words_UNK"])
#     return examples


# def encode_answer(examples, ans_to_aid):
#     # print('Warning: aid of answer not in vocab is -1')
#     for i, ex in enumerate(examples):
#         ex["answer_aid"] = ans_to_aid.get(
#             ex["answer"], -1
#         )  # -1 means answer not in vocab
#     return examples


# def encode_answers_occurence(examples, ans_to_aid):
#     for i, ex in enumerate(examples):
#         answers = []
#         answers_aid = []
#         answers_count = []
#         for ans in ex["answers_occurence"]:
#             aid = ans_to_aid.get(ans[0], -1)  # -1 means answer not in vocab
#             if aid != -1:
#                 answers_aid.append(aid)
#                 answers_count.append(ans[1])
#                 answers.append(ans[0])  # to store the original answer
#         ex["answers"] = answers
#         ex["answers_aid"] = answers_aid
#         ex["answers_count"] = answers_count
#     return examples


# def vqa_processed(params):

#     #####################################################
#     ## Read input files
#     #####################################################

#     interim_subfolder = (
#         "selected_interim"
#         if "select_questions" in params.keys() and params["select_questions"]
#         else "interim"
#     )
#     path_train = os.path.join(
#         params["dir"],
#         interim_subfolder,
#         params["trainsplit"] + "_questions_annotations.json",
#     )
#     if params["trainsplit"] == "train":
#         path_val = os.path.join(
#             params["dir"], interim_subfolder, "val_questions_annotations.json"
#         )
#     path_test = os.path.join(
#         params["dir"], interim_subfolder, "test_questions.json"
#     )
#     path_testdev = os.path.join(
#         params["dir"], interim_subfolder, "testdev_questions.json"
#     )

#     # An example is a tuple (question, image, answer)
#     # /!\ test and test-dev have no answer
#     trainset = json.load(open(path_train, "r"))
#     if params["trainsplit"] == "train":
#         valset = json.load(open(path_val, "r"))
#     testset = json.load(open(path_test, "r"))
#     testdevset = json.load(open(path_testdev, "r"))

#     #####################################################
#     ## Preprocess examples (questions and answers)
#     #####################################################

#     top_answers = get_top_answers(trainset, params["nans"])
#     aid_to_ans = [a for i, a in enumerate(top_answers)]
#     ans_to_aid = {a: i for i, a in enumerate(top_answers)}
#     # Remove examples if answer is not in top answers
#     trainset = remove_examples(trainset, ans_to_aid)

#     # Add 'question_words' to the initial tuple
#     trainset = preprocess_questions(trainset, params["nlp"])
#     trainset = extract_concepts(trainset)
#     cid_to_concept, concept_to_cid = buildup_concept_vocab(
#         trainset, mincount=params["concept_mincount"]
#     )
#     trainset = encode_concepts(trainset, cid_to_concept, concept_to_cid)
#     if params["trainsplit"] == "train":
#         valset = preprocess_questions(valset, params["nlp"])
#         # To validate the question generation performance, we remove the examples which have no answers
#         valset = remove_examples(valset, ans_to_aid)
#         valset = extract_concepts(valset)
#         valset = encode_concepts(valset, cid_to_concept, concept_to_cid)
#     testset = preprocess_questions(testset, params["nlp"])
#     testdevset = preprocess_questions(testdevset, params["nlp"])

#     # Also process top_words which contains a UNK char
#     trainset, top_words = remove_long_tail_train(trainset, params["minwcount"])

#     # The original code is from 1 without 'EOS'. Here we add 'EOS' as the index-0 word
#     # In addition, we also add 'START' in consideration of the performance
#     # print('Insert special [EOS] and [START] token')
#     top_words = (
#         ["EOS"] + top_words + ["START"]
#     )  # EOS should be at the beginning, otherwise error occurs
#     wid_to_word = {i: w for i, w in enumerate(top_words)}
#     word_to_wid = {w: i for i, w in enumerate(top_words)}

#     if params["trainsplit"] == "train":
#         valset = remove_long_tail_test(valset, word_to_wid)
#     testset = remove_long_tail_test(testset, word_to_wid)
#     testdevset = remove_long_tail_test(testdevset, word_to_wid)

#     trainset = encode_question(
#         trainset, word_to_wid, params["maxlength"], params["pad"]
#     )
#     if params["trainsplit"] == "train":
#         valset = encode_question(
#             valset, word_to_wid, params["maxlength"], params["pad"]
#         )
#     testset = encode_question(
#         testset, word_to_wid, params["maxlength"], params["pad"]
#     )
#     testdevset = encode_question(
#         testdevset, word_to_wid, params["maxlength"], params["pad"]
#     )

#     trainset = encode_answer(trainset, ans_to_aid)
#     trainset = encode_answers_occurence(trainset, ans_to_aid)
#     if params["trainsplit"] == "train":
#         valset = encode_answer(valset, ans_to_aid)
#         valset = encode_answers_occurence(valset, ans_to_aid)

#     #####################################################
#     ## Write output files
#     #####################################################

#     # Paths to output files
#     # Ex: data/vqa/processed/nans,3000_maxlength,15_..._trainsplit,train_testsplit,val/id_to_word.json
#     subdirname = "nans," + str(params["nans"])
#     for param in ["maxlength", "minwcount", "nlp", "pad", "trainsplit"]:
#         subdirname += "_" + param + "," + str(params[param])
#     if "select_questions" in params.keys() and params["select_questions"]:
#         subdirname += "_filter_questions"
#     os.system(
#         "mkdir -p " + os.path.join(params["dir"], "processed", subdirname)
#     )

#     path_wid_to_word = os.path.join(
#         params["dir"], "processed", subdirname, "wid_to_word.pickle"
#     )
#     path_word_to_wid = os.path.join(
#         params["dir"], "processed", subdirname, "word_to_wid.pickle"
#     )
#     path_cid_to_concept = os.path.join(
#         params["dir"], "processed", subdirname, "cid_to_concept.pickle"
#     )
#     path_concept_to_cid = os.path.join(
#         params["dir"], "processed", subdirname, "concept_to_cid.pickle"
#     )
#     path_aid_to_ans = os.path.join(
#         params["dir"], "processed", subdirname, "aid_to_ans.pickle"
#     )
#     path_ans_to_aid = os.path.join(
#         params["dir"], "processed", subdirname, "ans_to_aid.pickle"
#     )
#     if params["trainsplit"] == "train":
#         path_trainset = os.path.join(
#             params["dir"], "processed", subdirname, "trainset.pickle"
#         )
#         path_valset = os.path.join(
#             params["dir"], "processed", subdirname, "valset.pickle"
#         )
#     elif params["trainsplit"] == "trainval":
#         path_trainset = os.path.join(
#             params["dir"], "processed", subdirname, "trainvalset.pickle"
#         )
#     path_testset = os.path.join(
#         params["dir"], "processed", subdirname, "testset.pickle"
#     )
#     path_testdevset = os.path.join(
#         params["dir"], "processed", subdirname, "testdevset.pickle"
#     )

#     # print('Write wid_to_word to', path_wid_to_word)
#     with open(path_wid_to_word, "wb") as handle:
#         pickle.dump(wid_to_word, handle)

#     # print('Write word_to_wid to', path_word_to_wid)
#     with open(path_word_to_wid, "wb") as handle:
#         pickle.dump(word_to_wid, handle)

#     # print('Write cid_to_concept to', path_cid_to_concept)
#     with open(path_cid_to_concept, "wb") as handle:
#         pickle.dump(cid_to_concept, handle)

#     # print('Write concept_to_cid to', path_concept_to_cid)
#     with open(path_concept_to_cid, "wb") as handle:
#         pickle.dump(concept_to_cid, handle)

#     # print('Write aid_to_ans to', path_aid_to_ans)
#     with open(path_aid_to_ans, "wb") as handle:
#         pickle.dump(aid_to_ans, handle)

#     # print('Write ans_to_aid to', path_ans_to_aid)
#     with open(path_ans_to_aid, "wb") as handle:
#         pickle.dump(ans_to_aid, handle)

#     # print('Write trainset to', path_trainset)
#     with open(path_trainset, "wb") as handle:
#         pickle.dump(trainset, handle)

#     if params["trainsplit"] == "train":
#         # print('Write valset to', path_valset)
#         with open(path_valset, "wb") as handle:
#             pickle.dump(valset, handle)

#     # print('Write testset to', path_testset)
#     with open(path_testset, "wb") as handle:
#         pickle.dump(testset, handle)

#     # print('Write testdevset to', path_testdevset)
#     with open(path_testdevset, "wb") as handle:
#         pickle.dump(testdevset, handle)


def preprocess(train_file: str, out_dir: str, params: dict):
    trainset = json.load(open(train_file))
    trainset = [x for x in trainset if not x["need_external_knowledge"]]

    top_answers = get_top_answers(trainset, top_n=params["nans"])
    aid_to_ans = [a for i, a in enumerate(top_answers)]
    ans_to_aid = {a: i for i, a in enumerate(top_answers)}

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
@click.argument("train_file", type=click.Path(exists=True))
@click.argument("path_opt", type=click.Path(exists=True))
def main(train_file, path_opt):
    with open(path_opt, "r") as f:
        params = yaml.load(f)

    preprocess(train_file, params["dataset"]["dict_dir"], params["dataset"])


if __name__ == "__main__":
    main()
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--dirname",
#         default="data/vqa",
#         type=str,
#         help="Root directory containing raw, interim and processed directories",
#     )
#     parser.add_argument(
#         "--trainsplit",
#         default="train",
#         type=str,
#         help="Options: train | trainval",
#     )
#     parser.add_argument(
#         "--nans",
#         default=2000,
#         type=int,
#         help="Number of top answers for the final classifications",
#     )
#     parser.add_argument(
#         "--maxlength",
#         default=26,
#         type=int,
#         help="Max number of words in a caption. Captions longer get clipped",
#     )
#     parser.add_argument(
#         "--minwcount",
#         default=0,
#         type=int,
#         help="Words that occur less than that are removed from vocab",
#     )
#     parser.add_argument(
#         "--nlp",
#         default="mcb",
#         type=str,
#         help="Token method ; Options: nltk | mcb | naive",
#     )
#     parser.add_argument(
#         "--pad",
#         default="left",
#         type=str,
#         help="Padding ; Options: right (finish by zeros) | left (begin by zeros)",
#     )
#     args = parser.parse_args()
#     opt_vqa = vars(args)
#     vqa_processed(opt_vqa)

