from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchmetrics.functional import accuracy
from interpretable_resnet import get_resnet


class LightningResnet(LightningModule):
    def __init__(self, resnet_layers=18,  input_channels=3, num_classes=10, lr=0.001, batch_size=32, freeze=True):
        super(LightningResnet, self).__init__()
        self.save_hyperparameters()

        self.model = get_resnet(resnet_layers)

        trained_kernel = self.model.conv1.weight
        # replace the first conv layer
        self.model.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        if input_channels == 1:
            self.model.conv1.weight = nn.Parameter(
                trained_kernel.mean(dim=1, keepdim=True)
            )
        elif input_channels >= 3:
            pass
            #self.model.conv1.weight = nn.Parameter(trained_kernel[:, :3])

        if freeze:


            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        # for deepglobe:
        x = batch["image"]
        y = batch["mask"]
        #x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.hparams.num_classes
        )

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // self.hparams.batch_size
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


