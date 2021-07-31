"""Functions to handle the provided data.
"""

# Python standard imports
from typing import Optional, List, Callable, Tuple
from argparse import ArgumentParser
import os
from PIL import Image
import numpy as np

# Pytorch imports
import torch
import pytorch_lightning as pl
from torch.utils import data

# sklearn
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train', 'truck']

class FewShotDataset(data.Dataset):
    """Class representing dataset used for few shot learning task.
    It implements magic methods such as __getitem__ and __len__
    """
    def __init__(self, dataset: List, ops: Callable):
        """Initializing the FewShotDataset class with the required params.

        Args:
            dataset (List[int, str]): List containing a set (train or test) of
                paired (class_id, path_to_image) data.
            ops (Callable): if called it applies a list of transformations to
                the data. The required transformations depend on the CLIP method
                used as backbone.
        """
        self.dataset = dataset
        self.ops = ops

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Returns dataset item at a given 'idx'

        Args:
            idx (int): determines the item to get from the dataset.

        Returns:
            torch.Tensor: data ready to be used
        """
        _class, _img = self.dataset[idx]

        _img = Image.open(_img)
        _img = self.ops(_img)

        return _img, _class

    def __len__(self) -> int:
        return len(self.dataset)


class FewShotDataModule(pl.LightningDataModule):
    """Class encapsulating all the routines to handle FewShot Dataset.
    """
    def __init__(self, ops: Callable, batch_size: int = 4,
                 num_workers: int = 8, path_to_data: str = './dataset/few_shot/'):
        """Class encapsulating all the routines to handle FewShot Dataset.

        Args:
            ops (Callable): if called it applies a list of transformations to
                the data. The required transformations depend on the CLIP method
                used as backbone.
            batch_size (int, optional): indicates the number of samples per batch. Defaults to 4.
            num_workers (int, optional): indicate the number of threads created for the data 
                loading. Defaults to 8.
            path_to_data (str, optional): indicates the relative path to the dataset. Defaults
                to './dataset/few_shot/'.
        """
        super(FewShotDataModule, self).__init__()

        self.ops = ops
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.splits = {} # Contains train and valid splits.
        self.datasets = {} # Contains instances of the Dataset class. One per data spit.
        self.class_map = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))
        self.weights = [0] * len(CLASS_NAMES)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Specific arguments to the dataset and data loading process.

        Args:
            parent_parser (ArgumentParser): parent argument parser gathering all the parameters.

        Returns:
            ArgumentParser: parent argument parser with the new parameters.
        """

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--path_to_data', type=str, default='./dataset/few_shot/',
                            help='relative path to FewShot dataset. Defaults to ./dataset/few_shot/')
        parser.add_argument('--num_workers', type=int, default=8,
                            help='number of processes to handle data loading. Defaults to 8.')
        parser.add_argument('--batch_size', type=int, default=4,
                            help='number of samples per batch. Defaults to 4.')

        return parser


    def prepare_data(self, *args, **kwargs):
        """Data preparation routine. Reading the data from disk, 
        preparing splits, and so.
        """
        # get paths to train and test splits
        _split_paths = [os.path.join(self.path_to_data, split)
                            for split in os.listdir(self.path_to_data)]

        # for each split [train, test]
        for _path in _split_paths:
            _img_classes = os.listdir(_path) # get subfolders representing each class
            self.splits[os.path.basename(_path)] = []

            # get the images in pairs with its corresponding class
            for _class in _img_classes:
                _data = self.get_img_text_pair(os.path.join(_path, _class))

                if os.path.basename(_path) == 'train':
                    self.weights[self.encode_label(_class)] = len(_data)
                self.splits[os.path.basename(_path)].extend(_data)


    def encode_label(self, label: str) -> int:
        """Function used to encode the text label to a class index.

        Args:
            label (str): text label, e.g. airplane.

        Returns:
            int: class index according to the class position in CLASS_NAMES
        """
        return self.class_map[label]

    def get_img_text_pair(self, class_path: str) -> List:
        """Returns a list containint images of a given class and its corresponding class idx.

        Args:
            class_path (str): path to the folder class.

        Returns:
            List ([int, str]): list with pairs of [class index, path to image]. 
        """
        _class = self.encode_label(os.path.basename(class_path))
        _data = [[_class, os.path.join(class_path, i)]
                    for i in os.listdir(class_path) if i.endswith('.jpg')]

        return _data

    def setup(self, stage: Optional[str] = None):
        """Operations to be performed on each GPU.
        Here we build the datasets for the dataloaders. We also split
        the training set into train and validation in a stratified way
        (taking into account the class imbalance).

        Args:
            stage (Optional[str], optional): Training or test stage. Defaults to None.
        """
        if stage in (None, 'fit'):
            # Get a 20% of the train data for validation in a stratified way.
            _x = [i[1] for i in self.splits['train']]
            _y = [i[0] for i in self.splits['train']]

            _train_x, _val_x, _train_y, _val_y = train_test_split(_x, _y, test_size=0.2,
                                                                  stratify=_y)
            #print(np.unique(_train_y, return_counts=True))
            #print(np.unique(_val_y, return_counts=True))

            self.splits['train'] = [[i, j] for i,j in zip(_train_y, _train_x)]
            self.splits['valid'] = [[i, j] for i,j in zip(_val_y, _val_x)]

            self.datasets['train'] = FewShotDataset(self.splits['train'], self.ops)
            self.datasets['valid'] = FewShotDataset(self.splits['valid'], self.ops)

        if stage in (None, 'test'):
            self.datasets['test'] = FewShotDataset(self.splits['test'], self.ops)

    def train_dataloader(self) -> data.DataLoader:
        """Getter for the training dataloader

        Returns:
            data.DataLoader: train dataloader.
        """
        # Random weighted sampler to approach the imbalanced dataset
        self.weights = [1.0 / i for i in self.weights]

        _sample_weights = [0] * len(self.datasets['train'])

        for idx, (_, label) in enumerate(self.datasets['train']):
            _weight = self.weights[label]
            _sample_weights[idx] = _weight

        random_sampler = data.WeightedRandomSampler(_sample_weights,
                                                    len(self.datasets['train']), replacement=False)

        return data.DataLoader(dataset=self.datasets['train'], batch_size=self.batch_size,
                               num_workers=self.num_workers, pin_memory=False,
                               sampler=random_sampler)

    def val_dataloader(self) -> data.DataLoader:
        """Getter for the validation dataloader

        Returns:
            data.DataLoader: valid dataloader.
        """
        return data.DataLoader(dataset=self.datasets['valid'], batch_size=self.batch_size,
                               num_workers=self.num_workers, shuffle=False, pin_memory=False)

    def test_dataloader(self) -> data.DataLoader:
        """Getter for the test dataloader.

        Returns:
            data.DataLoader: test dataloader.
        """
        return data.DataLoader(dataset=self.datasets['test'], batch_size=self.batch_size,
                               num_workers=self.num_workers, shuffle=False, pin_memory=False)
