import os
from typing import List, Tuple, Union, Optional

# import torch
import torch.utils.data as data
import numpy as np
import pdb
import json

# from collections import defaultdict

from torch.utils.data import DataLoader
from .utils import AbstractVQADataset

# from .vqa_interim import vqa_interim
# from .vqa2_interim import vqa_interim as vqa2_interim

# from .vqa_processed import vqa_processed
# from .clevr_interim import clevr_interim
# from .clevr_processed import clevr_processed
# from . import coco
from .features import FeaturesDataset

# from . import vgenome
# from . import clevr

# import copy
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


# class VQA(AbstractVQA):
#     def _raw(self):
#         dir_zip = os.path.join(self.dir_raw, "zip")
#         dir_ann = os.path.join(self.dir_raw, "annotations")
#         os.system("mkdir -p " + dir_zip)
#         os.system("mkdir -p " + dir_ann)
#         os.system(
#             "wget https://vision.ece.vt.edu/vqa/release_data/mscoco/vqa/Questions_Train_mscoco.zip -P "
#             + dir_zip
#         )
#         os.system(
#             "wget https://vision.ece.vt.edu/vqa/release_data/mscoco/vqa/Questions_Val_mscoco.zip -P "
#             + dir_zip
#         )
#         os.system(
#             "wget https://vision.ece.vt.edu/vqa/release_data/mscoco/vqa/Questions_Test_mscoco.zip -P "
#             + dir_zip
#         )
#         os.system(
#             "wget https://vision.ece.vt.edu/vqa/release_data/mscoco/vqa/Annotations_Train_mscoco.zip -P "
#             + dir_zip
#         )
#         os.system(
#             "wget https://vision.ece.vt.edu/vqa/release_data/mscoco/vqa/Annotations_Val_mscoco.zip -P "
#             + dir_zip
#         )
#         os.system(
#             "unzip "
#             + os.path.join(dir_zip, "Questions_Train_mscoco.zip")
#             + " -d "
#             + dir_ann
#         )
#         os.system(
#             "unzip "
#             + os.path.join(dir_zip, "Questions_Val_mscoco.zip")
#             + " -d "
#             + dir_ann
#         )
#         os.system(
#             "unzip "
#             + os.path.join(dir_zip, "Questions_Test_mscoco.zip")
#             + " -d "
#             + dir_ann
#         )
#         os.system(
#             "unzip "
#             + os.path.join(dir_zip, "Annotations_Train_mscoco.zip")
#             + " -d "
#             + dir_ann
#         )
#         os.system(
#             "unzip "
#             + os.path.join(dir_zip, "Annotations_Val_mscoco.zip")
#             + " -d "
#             + dir_ann
#         )

#     def _interim(self, select_questions=False):
#         vqa_interim(self.opt["dir"], select_questions=select_questions)

#     def _processed(self):
#         vqa_processed(self.opt)


# class VQA2(AbstractVQA):
#     def _raw(self):
#         dir_zip = os.path.join(self.dir_raw, "zip")
#         dir_ann = os.path.join(self.dir_raw, "annotations")
#         os.system("mkdir -p " + dir_zip)
#         os.system("mkdir -p " + dir_ann)
#         os.system(
#             "wget https://vision.ece.vt.edu/vqa/release_data/mscoco/vqa/v2_Questions_Train_mscoco.zip -P "
#             + dir_zip
#         )
#         os.system(
#             "wget https://vision.ece.vt.edu/vqa/release_data/mscoco/vqa/v2_Questions_Val_mscoco.zip -P "
#             + dir_zip
#         )
#         os.system(
#             "wget https://vision.ece.vt.edu/vqa/release_data/mscoco/vqa/v2_Questions_Test_mscoco.zip -P "
#             + dir_zip
#         )
#         os.system(
#             "wget https://vision.ece.vt.edu/vqa/release_data/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P "
#             + dir_zip
#         )
#         os.system(
#             "wget https://vision.ece.vt.edu/vqa/release_data/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P "
#             + dir_zip
#         )
#         os.system(
#             "unzip "
#             + os.path.join(dir_zip, "v2_Questions_Train_mscoco.zip")
#             + " -d "
#             + dir_ann
#         )
#         os.system(
#             "unzip "
#             + os.path.join(dir_zip, "v2_Questions_Val_mscoco.zip")
#             + " -d "
#             + dir_ann
#         )
#         os.system(
#             "unzip "
#             + os.path.join(dir_zip, "v2_Questions_Test_mscoco.zip")
#             + " -d "
#             + dir_ann
#         )
#         os.system(
#             "unzip "
#             + os.path.join(dir_zip, "v2_Annotations_Train_mscoco.zip")
#             + " -d "
#             + dir_ann
#         )
#         os.system(
#             "unzip "
#             + os.path.join(dir_zip, "v2_Annotations_Val_mscoco.zip")
#             + " -d "
#             + dir_ann
#         )
#         os.system(
#             "mv "
#             + os.path.join(dir_ann, "v2_mscoco_train2014_annotations.json")
#             + " "
#             + os.path.join(dir_ann, "mscoco_train2014_annotations.json")
#         )
#         os.system(
#             "mv "
#             + os.path.join(dir_ann, "v2_mscoco_val2014_annotations.json")
#             + " "
#             + os.path.join(dir_ann, "mscoco_val2014_annotations.json")
#         )
#         os.system(
#             "mv "
#             + os.path.join(
#                 dir_ann, "v2_OpenEnded_mscoco_train2014_questions.json"
#             )
#             + " "
#             + os.path.join(dir_ann, "OpenEnded_mscoco_train2014_questions.json")
#         )
#         os.system(
#             "mv "
#             + os.path.join(
#                 dir_ann, "v2_OpenEnded_mscoco_val2014_questions.json"
#             )
#             + " "
#             + os.path.join(dir_ann, "OpenEnded_mscoco_val2014_questions.json")
#         )
#         os.system(
#             "mv "
#             + os.path.join(
#                 dir_ann, "v2_OpenEnded_mscoco_test2015_questions.json"
#             )
#             + " "
#             + os.path.join(dir_ann, "OpenEnded_mscoco_test2015_questions.json")
#         )
#         os.system(
#             "mv "
#             + os.path.join(
#                 dir_ann, "v2_OpenEnded_mscoco_test-dev2015_questions.json"
#             )
#             + " "
#             + os.path.join(
#                 dir_ann, "OpenEnded_mscoco_test-dev2015_questions.json"
#             )
#         )

#     def _interim(self, select_questions=False):
#         vqa2_interim(self.opt["dir"], select_questions=select_questions)

#     def _processed(self):
#         vqa_processed(self.opt)


# class VQAVisualGenome(data.Dataset):
#     def __init__(self, dataset_vqa, dataset_vgenome):
#         self.dataset_vqa = dataset_vqa
#         self.dataset_vgenome = dataset_vgenome
#         self._filter_dataset_vgenome()

#     def _filter_dataset_vgenome(self):
#         print("-> Filtering dataset vgenome")
#         data_vg = self.dataset_vgenome.dataset
#         ans_to_aid = self.dataset_vqa.ans_to_aid
#         word_to_wid = self.dataset_vqa.word_to_wid
#         data_vg_new = []
#         not_in = 0
#         for i in range(len(data_vg)):
#             if data_vg[i]["answer"] not in ans_to_aid:
#                 not_in += 1
#             else:
#                 data_vg[i]["answer_aid"] = ans_to_aid[data_vg[i]["answer"]]
#                 for j in range(data_vg[i]["seq_length"]):
#                     word = data_vg[i]["question_words_UNK"][j]
#                     if word in word_to_wid:
#                         wid = word_to_wid[word]
#                     else:
#                         wid = word_to_wid["UNK"]
#                     data_vg[i]["question_wids"][j] = wid
#                 data_vg_new.append(data_vg[i])
#         print("-> {} / {} items removed".format(not_in, len(data_vg)))
#         self.dataset_vgenome.dataset = data_vg_new
#         print(
#             "-> {} items left in visual genome".format(
#                 len(self.dataset_vgenome)
#             )
#         )
#         print("-> {} items total in vqa+vg".format(len(self)))

#     def __getitem__(self, index):
#         if index < len(self.dataset_vqa):
#             item = self.dataset_vqa[index]
#             # print('vqa')
#         else:
#             item = self.dataset_vgenome[index - len(self.dataset_vqa)]
#             # print('vg')
#         # import ipdb; ipdb.set_trace()
#         return item

#     def __len__(self):
#         return len(self.dataset_vqa) + len(self.dataset_vgenome)

#     def num_classes(self):
#         return self.dataset_vqa.num_classes()

#     def vocab_words(self):
#         return self.dataset_vqa.vocab_words()

#     def vocab_answers(self):
#         return self.dataset_vqa.vocab_answers()

#     def data_loader(self, batch_size=10, num_workers=4, shuffle=False):
#         return DataLoader(
#             self,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             pin_memory=False,
#         )

#     def split_name(self, testdev=False):
#         return self.dataset_vqa.split_name(testdev=testdev)


# class CLEVR(AbstractVQADataset):
#     def __init__(self, data_split, opt, dataset_img=None):
#         super(CLEVR, self).__init__(data_split, opt, dataset_img)

#         if (
#             "test" in self.data_split
#         ):  # means self.data_split is 'val' or 'test'
#             raise NotImplementedError

#     def _raw(self):
#         raise NotImplementedError

#     def _interim(self, select_questions=False):
#         clevr_interim(self.opt["dir"], select_questions=select_questions)

#     def _processed(self):
#         clevr_processed(self.opt)

#     def __getitem__(self, index):
#         item = {}
#         # TODO: better handle cascade of dict items
#         item_vqa = self.dataset[index]

#         # Process Visual (image or features)
#         if self.dataset_img is not None:
#             item_img = self.dataset_img.get_by_name(item_vqa["image_name"])
#             item["visual"] = item_img["visual"]
#             # Yikang added for tracing the image path
#             item["image"] = item_vqa["image_name"]

#         # Process Question (word token)
#         item["question_id"] = item_vqa["question_id"]
#         # Add additional <START> Token if set
#         question = (
#             [self.word_to_wid["START"]] + item_vqa["question_wids"]
#             if self.opt["add_start"]
#             else item_vqa["question_wids"]
#         )
#         item["question"] = torch.LongTensor(question)

#         if self.data_split == "test":
#             raise NotImplementedError
#         else:
#             # Process Answer if exists
#             item["answer"] = item_vqa["answer_aid"]

#             # if 'sample_concept' in self.opt.keys() and self.opt['sample_concept']:
#             #     item['concept'] = torch.zeros(len(self.cid_to_concept))
#             #     for t_aid in item_vqa['concepts_cid']:
#             #         item['concept'][t_aid] = 1 # use several-hots vectors to indicate which answers is sampled

#         return item

#     def __len__(self):
#         if self.data_split == "train":
#             if self.opt.get("partial", -1) > 0:
#                 return int(len(self.dataset)) * self.opt["partial"]
#         return len(self.dataset)

#     def num_classes(self):
#         return len(self.aid_to_ans)

#     def vocab_words(self):
#         return list(self.wid_to_word.values())

#     def vocab_answers(self):
#         return self.aid_to_ans

#     def data_loader(self, batch_size=10, num_workers=4, shuffle=False):
#         return DataLoader(
#             self,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             pin_memory=True,
#             drop_last=False,
#         )

#     def split_name(self, testdev=False):
#         if testdev:
#             return "test-dev"
#         if self.data_split in ["train", "val"]:
#             return self.data_split
#         elif self.data_split == "test":
#             return self.data_split
#         elif self.data_split == "testdev":
#             return "test-dev"
#         else:
#             assert False, "Wrong data_split: {}".format(self.data_split)


def get_image_dataset(feat_file: str, metadata: str, mode: str):
    return FeaturesDataset(feat_file, metadata, mode)


def factory(dataset_path: str, split: str, opt: dict) -> data.Dataset:

    image_dataset = get_image_dataset(
        opt["feature"], opt["metadata"], opt["mode"]
    )

    dataset_vqa = VQADataset(dataset_path, split, opt, image_dataset)

    return dataset_vqa
