import os
from typing import List
import torch.utils.data as data
import numpy as np
import json
from torch.utils.data import DataLoader
from .features import FeaturesDataset
import pickle


class VQADataset(data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: str,
        opt: dict,
        dataset_img=None,
        ans_ignore_idx=None,
    ):
        self.load_dict(opt["dict_dir"])
        self.dataset_img = dataset_img

        if ans_ignore_idx is None:
            self.ans_ignore_idx = len(self.ans_to_aid)
        else:
            self.ans_ignore_idx = ans_ignore_idx

        self.dataset_path = dataset_path

        self.load_dataset(dataset_path, split)

    def load_dataset(self, dataset_path: str, split: str):
        with open(dataset_path) as f:
            dataset = json.load(f)

        # get max length of question
        q_len_max = 0
        for item in dataset:
            question = item["question"]
            question = question.split()
            question = [
                self.word_to_wid[word]
                for word in question
                if word in self.word_to_wid
            ]
            q_len_max = max(len(question), q_len_max)

        self.dataset: List = []
        q_len_max = 0
        for item in dataset:
            if split in ["train", "val"]:
                if item["need_external_knowledge"]:
                    continue
            question, answer, image = (
                item["question"],
                item["answer"],
                item["image"],
            )
            question = question.split()
            question = [
                self.word_to_wid[word]
                for word in question
                if word in self.word_to_wid
            ]
            q_len_max = max(len(question), q_len_max)

            # filter training data
            if split == "train":

                # remove OoV answers
                if not answer in self.ans_to_aid:
                    continue

            self.dataset.append([image, answer, question])

        for i in range(len(self.dataset)):
            question = self.dataset[i][2]
            q_len = len(question)
            question += [0] * (q_len_max - q_len)
            question = np.array(question)
            self.dataset[i][2] = question

    def load_dict(self, dict_dir: str):

        path_wid_to_word = os.path.join(dict_dir, "wid_to_word.pkl")
        path_word_to_wid = os.path.join(dict_dir, "word_to_wid.pkl")
        path_aid_to_ans = os.path.join(dict_dir, "aid_to_ans.pkl")
        path_ans_to_aid = os.path.join(dict_dir, "ans_to_aid.pkl")

        with open(path_wid_to_word, "rb") as handle:
            self.wid_to_word = pickle.load(handle)

        with open(path_word_to_wid, "rb") as handle:
            self.word_to_wid = pickle.load(handle)

        with open(path_aid_to_ans, "rb") as handle:
            self.aid_to_ans = pickle.load(handle)

        with open(path_ans_to_aid, "rb") as handle:
            self.ans_to_aid = pickle.load(handle)

    def __getitem__(self, index):
        item = {}
        # TODO: better handle cascade of dict items
        item_vqa = self.dataset[index][0]

        # Process Visual (image or features)
        if self.dataset_img is not None:
            item_img = self.dataset_img.get_by_name(item_vqa)
            item["visual"] = item_img["visual"]
            item["image"] = item_vqa  # Yikang added for tracing the image path
        answer = self.dataset[index][1].lower()
        # return ignored index when answer is out of vocabulary
        if answer in self.ans_to_aid:
            item["answer"] = self.ans_to_aid[answer]
        else:
            item["answer"] = self.ans_ignore_idx
        item["question"] = self.dataset[index][2]
        return item

    def __len__(self):
        return len(self.dataset)

    def num_classes(self):
        return len(self.aid_to_ans)

    def vocab_words(self):
        return list(self.wid_to_word.values())

    def vocab_answers(self):
        return self.aid_to_ans

    def data_loader(
        self, batch_size: int = 10, num_workers: int = 4, shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )


def get_image_dataset(feat_file: str, metadata: str, mode: str):
    return FeaturesDataset(feat_file, metadata, mode)


def factory(dataset_path: str, split: str, opt: dict) -> data.Dataset:

    image_dataset = get_image_dataset(
        opt["feature"], opt["metadata"], opt["mode"]
    )

    dataset_vqa = VQADataset(dataset_path, split, opt, image_dataset)

    return dataset_vqa
