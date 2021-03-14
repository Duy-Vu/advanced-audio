#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Optional, Union, Dict
from pickle import load as pickle_load
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from file_io import get_files_from_dir_with_pathlib


__author__ = 'Duy Vu'
__docformat__ = 'reStructuredText'
__all__ = ['MyDataset']


class MyDataset(Dataset):
    def __init__(self,
                 data_dir: Union[str, Path],
                 data_parent_dir: Optional[str] = '',
                 key_features: Optional[str] = 'features',
                 key_class: Optional[str] = 'class') \
            -> None:
        """An example of an object of class torch.utils.data.Dataset

        :param data_dir: Directory to read data from.
        :type data_dir: str
        :param data_parent_dir: Parent directory of the data, defaults\
                                to ``.
        :type data_parent_dir: str
        :param key_features: Key to use for getting the features,\
                             defaults to `features`.
        :type key_features: str
        :param key_class: Key to use for getting the class, defaults\
                          to `class`.
        :type key_class: str
        :param load_into_memory: Load the data into memory? Default to True
        :type load_into_memory: bool
        """
        super().__init__()
        data_path = Path(data_parent_dir, data_dir)
        self.files = get_files_from_dir_with_pathlib(data_path)
        self.key_features = key_features
        self.key_class = key_class

        for i, a_file in enumerate(self.files):
            self.files[i] = self._load_file(a_file)


    @staticmethod
    def _load_file(file_path: Path) \
            -> Dict[str, Union[int, np.ndarray]]:
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
        :rtype: (np.ndarray, int)
        """
        the_item = self.files[item]
        return the_item[self.key_features], the_item[self.key_class]


def getting_dataset(data_dir, splits=(0.85, 0.15)):
    """
    Getting dataset split
    """
    dataset = MyDataset(data_dir=Path(data_dir))
    
    # Split data
    train, valid = splits
    num_train = int(dataset.__len__() * train)
    num_valid = dataset.__len__() - num_train

    return random_split(dataset=dataset, 
                        lengths=[num_train, num_valid], 
                        generator=torch.Generator().manual_seed(42))


def getting_data_loader(data_name: str,
                 dataset: MyDataset,
                 batch_size) -> DataLoader:
    """
    Getting data loader
    """
    return DataLoader(dataset,
                      batch_size=batch_size,
                      drop_last=(data_name == "training"),
                      shuffle=(data_name != "testing"),
                      num_workers=4), \
           dataset.__len__() if data_name != "training" \
                             else batch_size*(dataset.__len__() // batch_size)


if __name__ == "__main__":
    data_dir = "log_mel"
    training_data, validation_data = getting_dataset(data_dir)

    batch_size = 32
    training_loader, num_train = getting_data_loader("training", training_data, batch_size)
    validation_loader, num_valid = getting_data_loader("validation", validation_data, batch_size)
    testing_loader, num_test = getting_data_loader("testing", MyDataset(data_dir='testing'), batch_size)
    print(num_train, num_valid, num_test)
    print()
    print(f"{len(training_loader)}, train")
    print(f"{len(training_loader.dataset)}, train")
    for data in training_loader:
        x, y = data
        print(f'x type: {type(x)} | y type: {type(y)}')
        print(f'x size: {x.size()} | y size: {y.size()}') 
    print(f"{len(validation_loader)}, test")
    print(f"{len(validation_loader.dataset)}, valid")
    """ for data in validation_loader:
        x, y = data
        print(f'x type: {type(x)} | y type: {type(y)}')
        print(f'x size: {x.size()} | y size: {y.size()}') """
    print(f"{len(testing_loader)}, test")
    print(f"{len(testing_loader.dataset)}, test")
    """ for data in testing_loader:
        x, y = data
        print(f'x type: {type(x)} | y type: {type(y)}')
        print(f'x size: {x.size()} | y size: {y.size()}')  """

# EOF