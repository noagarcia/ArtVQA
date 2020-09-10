from dataclasses import dataclass
from typing import Callable, Any, Optional
from PIL.Image import Image
import os
import pdb

# from mpi4py import MPI
import torch.utils.data as data
import torchvision.transforms as transforms

from .images import ImagesFolder, AbstractImagesDataset, default_loader
from .features import FeaturesDataset

COCO_TRAIN_URL = "http://msvocds.blob.core.windows.net/coco2014/train2014.zip"
COCO_VAL_URL = "http://msvocds.blob.core.windows.net/coco2014/val2014.zip"
COCO_TEST_URL = "http://msvocds.blob.core.windows.net/coco2015/test2015.zip"


@dataclass
class COCOImages(AbstractImagesDataset):
    # def __init__(
    #     self,
    #     data_split: str,
    #     opt: dict,
    #     transform: Optional[Callable[[Image], Image]],
    #     loader: Callable[[Any], Image] = default_loader,
    # ) -> None:
    #     super(COCOImages, self).__init__(data_split, opt, transform, loader)
    #     self.split_name = split_name(self.data_split)
    #     self.dir_split = os.path.join(self.dir_raw, self.split_name)

    def __post_init__(self):
        self.dataset = ImagesFolder(
            self.dir_split, transform=self.transform, loader=self.loader
        )
        self.name_to_index = self._load_name_to_index()

    def _download_dataset(self, rawdata_dir) -> None:
        if self.data_split == "train":
            download_url = COCO_TRAIN_URL
        elif self.data_split == "val":
            download_url = COCO_VAL_URL
        elif self.data_split == "test":
            download_url = COCO_TEST_URL
        else:
            assert False, f"data_split {self.data_split} not exists"

        os.system(f"wget {download_url} -P {rawdata_dir}")

        zip_file = os.path.join(rawdata_dir, self.split_name + ".zip")
        os.system(f"unzip {zip_file} -d {rawdata_dir}")

    def _load_name_to_index(self):
        self.name_to_index = {
            name: index for index, name in enumerate(self.dataset.imgs)
        }
        return self.name_to_index

    def __getitem__(self, index):
        item = self.dataset[index]
        item["name"] = os.path.join(self.split_name, item["name"])
        return item

    def __len__(self):
        return len(self.dataset)


class COCOTrainval(data.Dataset):
    def __init__(self, trainset, valset):
        self.trainset = trainset
        self.valset = valset

    def __getitem__(self, index):
        if index < len(self.trainset):
            item = self.trainset[index]
        else:
            item = self.valset[index - len(self.trainset)]
        return item

    def get_by_name(self, image_name):
        if image_name in self.trainset.name_to_index:
            index = self.trainset.name_to_index[image_name]
            item = self.trainset[index]
            return item
        elif image_name in self.valset.name_to_index:
            index = self.valset.name_to_index[image_name]
            item = self.valset[index]
            return item
        else:
            raise ValueError

    def __len__(self):
        return len(self.trainset) + len(self.valset)


def default_transform(size):
    transform = transforms.Compose(
        [
            transforms.Scale(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # resnet imagnet
            ),
        ]
    )
    return transform


def factory(
    data_split: str, opt: dict, transform: Optional[Callable[[Image], Image]] = None
) -> data.Dataset:
    if data_split == "trainval":
        trainset = factory("train", opt, transform)
        valset = factory("val", opt, transform)
        return COCOTrainval(trainset, valset)
    elif data_split in ["train", "val", "test"]:
        if opt["mode"] == "img":
            if transform is None:
                transform = default_transform(opt["size"])
            return COCOImages(data_split, opt["dir"] + "raw", transform)
        elif opt["mode"] in ["noatt", "att"]:
            return FeaturesDataset(opt["mode"])
        else:
            raise ValueError
    else:
        raise ValueError

