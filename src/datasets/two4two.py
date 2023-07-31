import os
import tarfile
from pathlib import Path

import pandas as pd
import pytorch_lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms


# Note - you must have torchvision installed for this example


class Two4TwoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_input_dir,
                 mode='train',
                 transform=None,
                 target_transform=None):

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

        img_name = os.path.join(self.root_dir,
                                self.parameters.iloc[idx, self.id_col_idx] + '.png')
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        mask_name = os.path.join(self.root_dir,
                                 self.parameters.iloc[idx, self.id_col_idx] + '_mask.png')
        mask = transforms.ToTensor()(Image.open(mask_name).convert('RGB'))

        label = self.parameters.iloc[idx, self.label_col_idx]

        sample = (image, mask, label)

        return sample

    def create_df(self, mode):

        label_data = pd.read_json(self.parameters_file, lines=True)
        label_data['label'] = label_data['obj_name'].apply(
            lambda x: 0 if x == 'stretchy' else 1)

        return label_data


class Two4TwoDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/two4two",
                 work_dir: str = None, batch_size: int = 32, resize: int = 128, transform=None, mask_transform=None):
        super().__init__()
        self.two2two__predict = None
        self.two2two_val = None
        self.two2two_test = None
        self.two2two_train = None
        self.data_dir = data_dir
        self.working_path = work_dir
        self.batch_size = batch_size
        self.num_workers = 0
        self.normalize_dict = {'mean': 0.1307, 'std': 0.3081}
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,)),
                                                 transforms.Resize(resize, antialias=True)
                                                 ])
        if mask_transform:
            self.mask_transform = mask_transform

    def prepare_data(self):
        # extract ?
        file = Path(self.data_dir)
        if file.is_dir():
            return
        elif tarfile.is_tarfile(file):

            tar = tarfile.open(self.data_dir, "r")
            print(f"Found tarfile at {self.data_dir}")
            if self.working_path is None:
                # remove file ending from data_dir
                self.data_dir = os.path.splitext(self.data_dir)[0]
                tar.extractall(path=self.data_dir)
                print(f"Unpack at  {self.data_dir}")
            else:
                tar.extractall(path=self.working_path)
                print(f"Unpack at  {self.working_path}")
            tar.close()
        else:
            raise ValueError("Data directory is not a tarfile or directory")

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.two2two_train = Two4TwoDataset(self.data_dir, mode='train', transform=self.transform)
            self.two2two_val = Two4TwoDataset(self.data_dir, mode='validation', transform=self.transform)
        if stage == "validation":
            self.two2two_val = Two4TwoDataset(self.data_dir, mode='validation', transform=self.transform)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.two2two_test = Two4TwoDataset(self.data_dir, mode='test', transform=self.transform)

        if stage == "predict":
            self.two2two__predict = Two4TwoDataset(self.data_dir, mode='test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.two2two_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          drop_last=True, )

    def val_dataloader(self):
        return DataLoader(self.two2two_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.two2two_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.two2two_test, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == '__main__':
    path = "/home/jonasklotz/Downloads/two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15"
    datamodule = Two4TwoDataModule(path)
