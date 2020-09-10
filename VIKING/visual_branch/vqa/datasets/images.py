from typing import Optional, Callable
from dataclasses import dataclass
import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    "jpg",
    "JPG",
    "jpeg",
    "JPEG",
    "png",
    "PNG",
    "ppm",
    "PPM",
    "bmp",
    "BMP",
]


def is_image_file(filename: str) -> bool:
    return filename.split(".")[-1] in IMG_EXTENSIONS


def get_image_files(dir: str) -> list:
    images = []
    for fname in os.listdir(dir):
        if is_image_file(fname):
            images.append(fname)
    return images


def default_loader(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@dataclass
class ImagesFolder(data.Dataset):
    root: str
    transform: Optional[Callable] = None
    loader: Callable[[str], Image.Image] = default_loader

    def __post_init__(self):
        self.imgs = get_image_files(self.root)
        if len(self.imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in subfolders of: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )

    def __getitem__(self, index):
        item = {}
        item["name"] = self.imgs[index]
        item["path"] = os.path.join(self.root, item["name"])
        if self.loader is not None:
            item["visual"] = self.loader(item["path"])
            if self.transform is not None:
                item["visual"] = self.transform(item["visual"])
        return item

    def __len__(self):
        return len(self.imgs)


@dataclass
class AbstractImagesDataset(data.Dataset):
    data_split: str
    rawdata_dir: str
    transform: Optional[Callable] = None
    loader: Callable[[str], Image.Image] = default_loader

    def __post_init__(self):
        if not os.path.exists(self.rawdata_dir):
            self._download_dataset(self.rawdata_dir)

    def get_by_name(self, image_name):
        index = self.name_to_index[image_name]
        return self[index]

    def _raw(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
