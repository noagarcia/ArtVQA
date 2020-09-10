from dataclasses import dataclass
import os
from typing import Tuple
import h5py
import torch
import torch.utils.data as data


@dataclass
class FeaturesDataset(data.Dataset):
    feat_file: str
    metadata: str
    mode: str
    dir_extract: str = "data/SemArt/extract/arch,resnet152_size,448/"
    data_split: str = "all"

    def __post_init__(self) -> None:
        hdf5_file = h5py.File(self.feat_file, "r")
        self.dataset_features = hdf5_file[self.mode]
        self.index_to_name, self.name_to_index = self.image_index_mapping()

    def image_index_mapping(self) -> Tuple[list, dict]:
        with open(self.metadata, "r") as f:
            index_to_name = [line.rstrip() for line in f]
        name_to_index = {
            name: index for index, name in enumerate(index_to_name)
        }
        return index_to_name, name_to_index

    def __getitem__(self, index: int) -> dict:
        return {
            "name": self.index_to_name[index],
            "visual": self.get_features(index),
        }

    def get_features(self, index: int) -> torch.Tensor:
        return torch.Tensor(self.dataset_features[index])

    def get_by_name(self, image_name: str) -> dict:
        index = self.name_to_index[image_name]
        return self[index]

    def __len__(self) -> int:
        return self.dataset_features.shape[0]

