import os

import lightning as L
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

from models.VQVAE import VQVAE


def load_image_from_datamodule(datamodule, index=None):
    """Load an image from the datamodule

    :param datamodule: the datamodule
    :param index: the index of the image to load
    :param quant_norm: whether to normalize
    :return: image and label
    """

    datamodule.prepare_data()
    datamodule.setup(stage='validate')
    testloader = datamodule.val_dataloader()

    n_samples = len(testloader)
    if index is None:
        index = int(np.random.random() * n_samples)

    subset_indices = [index]  # select your indices here as a list
    subset = torch.utils.data.Subset(testloader.dataset, subset_indices)
    testloader_subset = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)

    # get the first batch
    batch = next(iter(testloader_subset))
    batch['image'] = batch['image'].squeeze()

    # get the first image
    img = batch['image']
    # get the first label
    label = batch['label']

    img = img.numpy()

    return img, label


def get_train_images_from_dataloader(train_dataset, num):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)


class GenerateCallback(Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                loss, reconst_imgs, perplexity  = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


def train_vqvae_dataloader(working_path, train_loader, val_loader, test_loader, max_epochs=50, learning_rate = 1e-3, decay = 0.99, commitment_cost = 0.25, embedding_dim = 16,  # 64,
                      num_residual_layers = 2, num_residual_hiddens = 32, num_hiddens = 128, num_training_updates = 15000, batch_size = 256 ):
    """ Trains a VQ-VAE given dataloader.


    Returns:
        model: Trained VQ-VAE model
        result: Dictionary containing test and validation metrics
    """
    # We use the hyperparameters from the author's code:

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=os.path.join(working_path, "vqvae_results"),
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            GenerateCallback(get_train_images_from_dataloader(train_loader, num=8), every_n_epochs=10),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_filename = os.path.join(CHECKPOINT_LOAD_PATH, "cifar10_%i.ckpt" % latent_dim)
    # if os.path.isfile(pretrained_filename):
    #    print("Found pretrained model, loading...")
    #    model = Autoencoder.load_from_checkpoint(pretrained_filename)
    # else:

    model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim,
                  commitment_cost, decay)

    trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result

batch_size = 256
num_training_updates = 15000
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 16  # 64
num_embeddings = 128  # 512
commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-3


def train_vqvae_datamodule(working_path, datamodule, max_epochs=50, learning_rate = 1e-3, decay = 0.99, commitment_cost = 0.25, embedding_dim = 16,  # 64,
                      num_residual_layers = 2, num_residual_hiddens = 32, num_hiddens = 128, num_training_updates = 15000, batch_size = 256 ):
    """ Trains a VQ-VAE given dataloader.


    Returns:
        model: Trained VQ-VAE model
        result: Dictionary containing test and validation metrics
    """

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=os.path.join(working_path, "vqvae_results"),
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            #GenerateCallback(load_image_from_datamodule(train_loader, num=8), every_n_epochs=10),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need



    model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim,
                  commitment_cost, decay)

    trainer.fit(model, datamodule=datamodule)
    # Test best model on validation and test set
    val_result = trainer.test(model, datamodule=datamodule, verbose=False)
    test_result = trainer.test(model, datamodule=datamodule, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result


if __name__ == '__main__':
    from datasets.two4two import Two4TwoDataModule
    working_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results"
    datapath = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/two4two2.tar.xz"
    datamodule = Two4TwoDataModule(datapath)

    train_vqvae_datamodule(working_path=working_path, datamodule=datamodule, max_epochs=1)
