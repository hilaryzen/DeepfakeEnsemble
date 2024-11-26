import os

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np

from PIL import Image


Split = Literal["train", "valid", "test"]


@dataclass
class PathDataset:
    path_images: Path
    split: Split

    def __post_init__(self):
        self.files = self.load_filelist(self.split)

    def load_filelist(self, split):
        return os.listdir(self.path_images / split)

    def get_file_name(self, i):
        return Path(self.files[i]).stem

    def get_image_path(self, i):
        return str(self.path_images / self.split / self.files[i])

    def __len__(self):
        return len(self.files)


class WithMasksPathDataset(PathDataset):
    def __init__(self, path_images, path_masks, split):
        self.path_masks = path_masks
        super().__init__(path_images, split)

    def get_mask_path(self, i):
        return str(self.path_masks / self.split / self.files[i])

    @staticmethod
    def load_mask_keep(path):
        """Assumes that the in the original mask:

        - `255` means unchanged content, and
        - `0` means modified content.

        """
        mask = np.array(Image.open(path))
        mask = 1 - (mask[:, :, 0] == 255).astype("float")
        return mask

    def load_mask(self, i):
        return self.load_mask_keep(self.get_mask_path(i))

    def __len__(self):
        return len(self.files)


# § · Real datasets


class IndividualizedDataset(PathDataset):
    def __init__(self, split):
        path_images = Path("data/individualized/real")
        super().__init__(path_images=path_images, split=split)

class DFFDataset(PathDataset):
    def __init__(self, split):
        path_images = Path("data/dff/real")
        super().__init__(path_images=path_images, split=split)

class ItauDataset(PathDataset):
    def __init__(self, split):
        path_images = Path("data/itau/real")
        super().__init__(path_images=path_images, split=split)


# § · Fake datasets


class IndividualizedFakeDataset(PathDataset):
    def __init__(self, split):
        path_images = Path("data/individualized/df")
        super().__init__(path_images=path_images, split=split)

class DFFFakeDataset(PathDataset):
    def __init__(self, split):
        path_images = Path("data/dff/df")
        super().__init__(path_images=path_images, split=split)

class ItauFakeDataset(PathDataset):
    def __init__(self, split):
        path_images = Path("data/itau/df")
        super().__init__(path_images=path_images, split=split)