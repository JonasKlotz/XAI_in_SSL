from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils

from datasets.two4two import Two4TwoDataModule

num_workers = 8
batch_size = 256
seed = 1
max_epochs = 2
input_size = 128
num_ftrs = 32

pl.seed_everything(seed)


class SimCLRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def shared_step(self, batch):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss_ssl", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]




if __name__ == '__main__':
    transform = SimCLRTransform(
        input_size=input_size, vf_prob=0.5, rr_prob=0.5, cj_prob=0.0, random_gray_scale=0.0
    )

    data_module = Two4TwoDataModule(batch_size=batch_size,
                                    transform=transform)
    from models.bolts import setup_model

    path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results/models/two4two/full_model3.pth"
    model, encoder, layers,_ = setup_model('simclr_pretrained', model_path=path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator=device)
    trainer.fit(model, data_module)
