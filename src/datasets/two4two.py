import os
import os
import tarfile
from pathlib import Path

import cv2
import lightning as L
import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader
# Note - you must have torchvision installed for this example
from torchvision import transforms


class Two4TwoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_input_dir,
                 mode='train',
                 transform=None,
                 target_transform=None):

        if transform is None:
            transform = transforms.ToTensor()

        self.root_dir = os.path.join(data_input_dir, mode)
        self.parameters_file = os.path.join(self.root_dir, 'parameters.jsonl')

        self.parameters = self.create_df(mode)
        self.id_col_idx = self.parameters.columns.get_loc("id")
        self.label_col_idx = self.parameters.columns.get_loc("label")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        intern_path = self.parameters.iloc[idx, self.id_col_idx]
        img_name = os.path.join(self.root_dir, intern_path + '.png')

        image = cv2.imread(str(img_name))
        mask_name = os.path.join(self.root_dir, intern_path + '_mask.png')
        mask = cv2.imread(str(mask_name))
        # binarize mask with open cv thresh

        mask = self.target_transform(mask)
        # convert into 1 channel

        # binarize mask
        # mask = torch.where(mask > 0, torch.tensor(1), torch.tensor(0))

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA) # convert to 3 channels

        label = self.parameters.iloc[idx, self.label_col_idx]

        image = self.transform(image)

        sample = (image / 255., mask, label)

        return sample

    def create_df(self, mode):

        label_data = pd.read_json(self.parameters_file, lines=True)
        label_data['label'] = label_data['obj_name'].apply(
            lambda x: 0 if x == 'sticky' else 1)
        # drop everything but id and label
        label_data = label_data[['id', 'label']]

        return label_data


class Two4TwoDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/two4two", working_path: str = None, batch_size: int = 32, resize: int = None):
        super().__init__()
        self.two2two__predict = None
        self.two2two_val = None
        self.two2two_test = None
        self.two2two_train = None
        self.data_dir = data_dir
        self.working_path = working_path
        self.batch_size = batch_size
        self.num_workers = 0
        normal_transforms = [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        if resize:
            normal_transforms += [transforms.Resize(resize)]
        self.transform = transforms.Compose(normal_transforms)
        mask_transforms = [transforms.ToPILImage(), transforms.ToTensor()]
        if resize:
            mask_transforms += [transforms.Resize(resize)]
        self.mask_transform = transforms.Compose(mask_transforms)

    def prepare_data(self):
        # extract ?
        file = Path(self.data_dir)
        if file.is_dir():
            return
        elif tarfile.is_tarfile(file):
            tar = tarfile.open(self.data_dir, "r")
            if self.working_path is None:
                # remove file ending from data_dir
                self.data_dir = os.path.splitext(self.data_dir)[0]
                tar.extractall(path=self.data_dir)
            else:
                tar.extractall(path=self.working_path)
            tar.close()
        else:
            raise ValueError("Data directory is not a tarfile or directory")

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.two2two_train = Two4TwoDataset(self.data_dir, mode='train', transform=self.transform,
                                                target_transform=self.mask_transform)
            self.two2two_val = Two4TwoDataset(self.data_dir, mode='validation', transform=self.transform,
                                              target_transform=self.mask_transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.two2two_test = Two4TwoDataset(self.data_dir, mode='test', transform=self.transform,
                                               target_transform=self.mask_transform)

        if stage == "predict":
            self.two2two__predict = Two4TwoDataset(self.data_dir, mode='test', transform=self.transform,
                                                   target_transform=self.mask_transform)

    def train_dataloader(self):
        return DataLoader(self.two2two_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.two2two_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.two2two_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.two2two_test, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == '__main__':
    path = "/home/jonasklotz/Downloads/two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15"
    datamodule = Two4TwoDataModule(path)
