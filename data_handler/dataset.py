#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path 
from typing import Tuple, Optional, Union, List
from pickle import load as pickle_load
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tools.file_io import get_files_from_dir_with_pathlib
import librosa

__docformat__ = 'reStructuredText'
__all__ = ['MyDataset']

scene_dict = {
    "airport": 0,
    "shopping_mall": 1,
    "metro_station": 2,
    "street_pedestrian": 3,
    "public_square": 4,
    "street_traffic": 5,
    "tram": 6,
    "bus": 7,
    "metro": 8,
    "park": 9
}
class ASDataset(Dataset):

    def __init__(self,
                 split: str,
                 load_into_memory: bool, 
                 data_features_dir: Union[str, Path],
                 data_parent_dir: Union[str, Path],
                 meta_parent_dir: Union[str, Path],
                 meta_dir: Union[str, Path],
                 normalize=True
                 ) \
            -> None:
        """An example of an object of class torch.utils.data.Dataset

        :param data_dir: Directory to read data from.
        :type data_dir: str
        :param data_parent_dir: Parent directory of the data, defaults\
                                to ``.
        :type data_parent_dir: str
        """
        super().__init__()
        self.normalize = normalize
        self.split = split
        data_path = Path(data_parent_dir, data_features_dir)
        self.files = get_files_from_dir_with_pathlib(data_path)

        if split != "test":
            self.df = pd.read_csv(Path(meta_parent_dir, meta_dir), sep="\t",index_col=["filename"]) 
            cpickle_path = [f'{f.split(".")[0].split("audio/")[1]}.cpickle' for f in self.df.index]
            self.files = [f for f in self.files if f.name in cpickle_path]

            audio_files = []
            self.scenes = []
            for f in self.files:
                audio_file = f'audio/{f.name.split(".")[0]}.wav' 
                self.scenes.append(scene_dict[self.df.loc[audio_file].values[0]])
        self.load_into_memory = load_into_memory
        self.items = [None for _ in range(len(self.files))]


        if self.load_into_memory:
            for i, a_file in enumerate(self.files):
                self.items[i] = self._load_file(a_file)

    @staticmethod
    def _load_file(file_path: Path) \
            -> List[np.ndarray]:
        """Loads a file using pathlib.Path

        :param file_path: File path.
        :type file_path: pathlib.Path
        :return: The file.
        :rtype: dict[str, int|np.ndarray]
        """
        with file_path.open('rb') as f:
            return pickle_load(f)

    def __len__(self) \
            -> int:
        """Returns the length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """
        return len(self.files)

    def __getitem__(self,
                    item: int) \
            -> Tuple[np.ndarray, int]:
        """Returns an item from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Features and class of the item.
        :rtype: (np.ndarray, np.ndarray)
        """
        if self.load_into_memory:
            feature = self.items[item]["_data"]
        else:
            feature = self._load_file(self.files[item])["_data"]
        #feature = librosa.power_to_db(feature).astype(np.float32)
        """
        feature = mono_to_color(feature)
        if self.normalize: 
            MEAN = np.array([0.485, 0.456, 0.406])
            STD = np.array([0.229, 0.224, 0.225])
            feature = normalize(feature, mean=MEAN, std=STD)
        """

        if self.split != "test":
            scene = self.scenes[item]
            return torch.Tensor(feature), torch.LongTensor([scene])
        else:
            return torch.Tensor(feature)


def get_data_loader(
                 dataset: ASDataset,
                 batch_size:int,
                 shuffle:bool, 
                 drop_last:bool,
                 num_workers) -> DataLoader:
    
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      num_workers=num_workers) 


def mono_to_color(X, eps=1e-6):
    X = np.stack([X, X, X], axis=-1)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(X, dtype=np.uint8)

    return V



def normalize(image, mean, std):
    image = (image / 255.0).astype(np.float32)
    image = (image - mean) / std
    return np.moveaxis(image, 2, 0)

# EOF

