import os
import sys
from typing import Optional

from visualization.plotting import plot_img_mask_heatmap
from xai.embedded_gradcam.database import build_database
from xai.embedded_gradcam.gradcam import generate_activations
from xai.embedded_gradcam.helpers import _plot_grad_heatmap
from xai.metrics.top_k_intersections import TopKIntersection


import numpy as np
from datasets.datautils import load_img_to_batch, extract_data_loader
import zarr
from datasets.two4two import Two4TwoDataModule
from models.VQVAE import VQVAE
import torch
from os import path

from skimage.transform import resize

# gradients = None
outer_activations = None

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


def read_database(database_path):
    gradients_zarr = path.join(database_path, "grad_array.zarr")
    embeddings_zarr = path.join(database_path, "embs_array.zarr")
    gradients = zarr.load(gradients_zarr)
    embeddings = zarr.load(embeddings_zarr)
    return embeddings, gradients


def get_k_nearest_neighbours(embeddings, embeddings_db, k=1):
    embeddings = embeddings.detach().cpu().numpy()

    # calculate the distance between the embeddings and the embeddings in the database
    distances = np.linalg.norm(embeddings_db - embeddings, axis=1)
    # sum the distances except for batch dimension
    distances = np.sum(distances, axis=tuple(range(1, distances.ndim)))
    # get the indices of the k smallest distances
    indices = np.argpartition(distances, k)[:k]
    return indices


def explain_image(image_tensor, model, layer, embeddings_db, gradients_db, k=5, plot=True):
    """Generates a heatmap for the given image tensor
    Args:
        image_tensor: the image tensor shape (b, c, h, w)
        model: the model
        layer: the layer to generate the heatmap from
        embeddings_db: the embeddings database
        gradients_db: the gradients database
        k: the number of nearest neighbours to use

    Returns:
        the heatmap as a numpy array with shape (h, w)
        """
    # add batch if necessary
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    dims = image_tensor.size()[2:]
    tmp_activations, embeddings = generate_activations(model, layer, image_tensor)
    # get gradients index from database via a k nearest neighbour search
    # get k nearest neighbours gradients
    index = get_k_nearest_neighbours(embeddings, embeddings_db, k)
    if isinstance(index, int):
        index = [index]

    # get the gradients from the database
    pooled_gradients = np.zeros(gradients_db[0].shape)
    for i in index:
        pooled_gradients += gradients_db[i]
    pooled_gradients /= len(index)

    # weight the channels by corresponding gradients
    for i in range(tmp_activations.size()[1]):
        tmp_activations[:, i, :, :] *= pooled_gradients[i]
    # average the channels of the activations
    heatmap = torch.mean(tmp_activations, dim=1).squeeze()

    # relu on top of the heatmap
    # heatmap = F.relu(heatmap) # why all negative

    # normalize the heatmap
    heatmap -= torch.min(heatmap)
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.detach().cpu().numpy()
    heatmap = resize(heatmap, dims)

    if plot:
        _plot_grad_heatmap(heatmap)

    return heatmap


def explain_batch(
        model: torch.nn.Module,
        layer: torch.nn.Module,
        embeddings_db: np.ndarray,
        gradients_db: np.ndarray,
        x_batch: np.ndarray,
        save: bool = True,
        save_path: Optional[os.PathLike] = None,
        plot: bool = False):
    """ Generates heatmaps for a batch of images

    Parameters
    ----------
    model: nn.Module
        The model to explain
    layer: nn.Module
        The layer to explain
    embeddings_db: np.ndarray
        The embeddings database
    gradients_db: np.ndarray
        The gradients database
    x_batch: np.ndarray
        The batch of images to explain
    save: bool
        Whether to save the heatmaps
    save_path: Optional[os.PathLike]
        The path to save the heatmaps to
    plot: bool
        Whether to plot the heatmaps
    Returns
    -------
    np.ndarray
        The heatmaps as a numpy array
    """

    heatmaps = np.zeros((x_batch.size()[0], x_batch.size()[2], x_batch.size()[3]))
    # iterate over batch and generate heatmaps
    for i in range(x_batch.size()[0]):
        heatmap = explain_image(x_batch[i], model, layer, embeddings_db, gradients_db, k=10, plot=plot)
        heatmaps[i] = heatmap

    if save:
        # save heatmaps as numpy array
        np.save(path.join(save_path, "a_batch.npy"), heatmaps)
    return heatmaps


if __name__ == '__main__':
    data_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/two4two"
    work_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results"
    model_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results/VAE.ckpt"
    database_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results/database"

    # img_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/test.png"
    # img_tensor = load_img_to_batch(img_path)
    model = VQVAE.load_from_checkpoint(model_path, map_location=device)
    data_module = Two4TwoDataModule(data_dir=data_path, working_path=work_path, batch_size=1)

    layer_name = '_pre_vq_conv'
    layer = getattr(model, layer_name)

    # build_database(data_module, model, work_path, layer, n=100)

    embeddings_db, gradients_db = read_database(database_path)

    # load batch of 64 images from test set
    data_module = Two4TwoDataModule(data_dir=data_path, working_path=work_path, batch_size=64)
    data_loader = extract_data_loader(data_module, "test")
    x_batch, s_batch, y_batch = next(iter(data_loader))

    # get heatmaps for batch
    a_batch = explain_batch(model, layer, embeddings_db, gradients_db, x_batch, save_path=work_path)

    # plot_img_mask_heatmap(x_batch[0], s_batch[0],  a_batch[0])

    # convert all to numpy
    # img = x_batch[0].detach().cpu().numpy()
    # label = y_batch[0].detach().cpu().numpy()
    # mask = s_batch[0].detach().cpu().numpy()
    # heatmap = a_batch[0]
    # eval = metric.evaluate_instance(model, img, label, heatmap, mask)

    metric = TopKIntersection(k=1000, return_aggregate=True)
    eval = metric(x_batch=x_batch, s_batch=s_batch, a_batch=a_batch)
    print(eval)
    print("done")


