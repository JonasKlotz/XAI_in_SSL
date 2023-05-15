import json
import optparse
import os
import time
import tarfile



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
import torch
from torchvision import models, transforms
import lightning as L

from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision import transforms

import cv2
from pathlib import Path


class Two4TwoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_input_dir,
                 mode='train',
                 transform=None):

        if transform is None:
            transform = transforms.ToTensor()

        self.root_dir = os.path.join(data_input_dir, mode)
        self.parameters_file = os.path.join(self.root_dir, 'parameters.jsonl')

        self.parameters = self.create_df(mode)
        self.id_col_idx = self.parameters.columns.get_loc("id")
        self.label_col_idx = self.parameters.columns.get_loc("label")

        self.transform = transform

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.parameters.iloc[idx, self.id_col_idx] + '.png')

        image = cv2.imread(str(img_name))
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA) # convert to 3 channels

        label = self.parameters.iloc[idx, self.label_col_idx]

        image = self.transform(transforms.ToPILImage()(image))

        sample = (image / 255., label)

        return sample

    def create_df(self, mode):

        label_data = pd.read_json(self.parameters_file, lines=True)
        label_data['label'] = label_data['obj_name'].apply(
            lambda x: 0 if x == 'sticky' else 1)

        return label_data


class Two4TwoDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.two2two__predict = None
        self.two2two_val = None
        self.two2two_test = None
        self.two2two_train = None
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # extract ?
        file = Path(self.data_dir)
        if file.is_dir():
            return
        elif tarfile.is_tarfile(file):
            tar = tarfile.open(self.data_dir, "r")
            # remove file ending from data_dir
            self.data_dir =os.path.splitext(self.data_dir)[0]
            tar.extractall(path=self.data_dir)
            tar.close()
        else:
            raise ValueError("Data directory is not a tarfile or directory")


    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.two2two_train = Two4TwoDataset(self.data_dir, mode='train', transform=self.transform)
            self.two2two_val = Two4TwoDataset(self.data_dir, mode='validation', transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.two2two_test = Two4TwoDataset(self.data_dir, mode='test', transform=self.transform)

        if stage == "predict":
            self.two2two__predict = Two4TwoDataset(self.data_dir, mode='test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.two2two_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.two2two_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.two2two_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.two2two_test, batch_size=32)


if __name__ == '__main__':
    path = "/home/jonasklotz/Downloads/two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15"
    datamodule = Two4TwoDataModule(path)
