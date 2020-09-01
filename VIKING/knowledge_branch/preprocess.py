from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
import re
import json


class Preprocessor(object):

    def stop_words_filter(self, words):
        stop_words = set(stopwords.words('english'))
        return [word for word in words if not word in stop_words]

    def stemming_words_filter(self, words):
        ps = PorterStemmer()
        return [ps.stem(word) for word in words]

    def preprocess(self, words, stop_words = True, stemming_words = True):
        filterd_result = words
        if stop_words:
            filterd_result = self.stop_words_filter(filterd_result)
        if stemming_words:
            filterd_result = self.stemming_words_filter(filterd_result)
        return filterd_result


def load_questions(filepath, preprocesser):
    result = []
    with open(filepath) as f:
        dataset = json.load(f)
    for item in dataset:
        image = item["image"]
        question_ = item["question"]
        answer_ = item["answer"]
        question = re.compile(r'\w+').findall(question_)
        question = [w for w in question if w != ""]
        result.append({"image":image, "question": question_, "answer": answer_, "original_question":question, "filtered_question":preprocessor.preprocess(question)})
    return result


def load_comments(directory, preprocesser):
    result = []
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), "r", encoding = "ISO-8859-1") as f:
            lines = f.readlines()
            for line in lines[1:]:
                items = line.split("\t")
                image, comment = items[0], items[1]
                comment = comment.lower()
                comment = re.compile(r'\w+').findall(comment)
                comment = [w for w in comment if w != ""]
                result.append({"image":image, "comment":comment, "filtered_comment":preprocessor.preprocess(comment)})
    return result


if __name__ == "__main__":
    preprocessor = Preprocessor()

    cache_dir = "Cache/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    questions = load_questions("../modality_selector/Results/test_need_kg.json", preprocessor)
    with open(os.path.join(cache_dir, "preprocessed_test_need_kg.json"), "w") as f:
        json.dump(questions, f, indent = 4)

    comments = load_comments("./comments/", preprocessor)
    with open(os.path.join(cache_dir, "preprocessed_comments.json"), "w") as f:
        json.dump(comments, f, indent = 4)

