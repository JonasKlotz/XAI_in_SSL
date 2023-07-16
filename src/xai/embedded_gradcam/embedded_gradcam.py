import sys

import warnings
from itertools import product

from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Adds the other directory to your python path.
sys.path.append("/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/src")
from general_utils import setup_paths, save_batches, parse_batch, read_batches

import os
from os import path
from typing import Optional

import numpy as np
import torch
from skimage.transform import resize

from visualization.plotting import plot_batches

from datasets.datautils import extract_data_loader, setup_datamodule
from models.bolts import setup_model
from xai.embedded_gradcam.database import read_database
from xai.embedded_gradcam.gradcam import generate_activations
from xai.embedded_gradcam.helpers import _plot_grad_heatmap

# gradients = None
outer_activations = None

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


def get_k_nearest_neighbours(embeddings, embeddings_db, k=1):
    embeddings = embeddings.detach().cpu().numpy()

    # calculate the distance between the embeddings and the embeddings in the database
    distances = np.linalg.norm(embeddings_db - embeddings, axis=1)
    # sum the distances except for batch dimension
    distances = np.sum(distances, axis=tuple(range(1, distances.ndim)))
    # get the indices of the k smallest distances
    indices = np.argpartition(distances, k)[:k]
    return indices


def explain_image(image_tensor, model, encoder, layer, embeddings_db, gradients_db, k=1, plot=True, save_path=None):
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

    model.eval()
    encoder.eval()

    # generate embeddings from image
    embeddings = encoder(image_tensor)
    if isinstance(embeddings, list):
        embeddings = embeddings[0]

    img_dims = image_tensor.size()[2:]
    tmp_activations = generate_activations(model, layer, image_tensor)
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
    heatmap = resize(heatmap, img_dims)

    if plot:
        _plot_grad_heatmap(heatmap, image_tensor=image_tensor, save_path=save_path)

    return heatmap


def explain_batch(
        model: torch.nn.Module,
        encoder: torch.nn.Module,
        layer: torch.nn.Module,
        embeddings_db: np.ndarray,
        gradients_db: np.ndarray,
        x_batch: np.ndarray,
        plot: bool = False,
        k: int = 1):
    """ Generates heatmaps for a batch of images

    Parameters
    ----------
    model: nn.Module
        The model to explain
    encoder: nn.Module
        The encoder to explain
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
    k: int
        The number of nearest neighbours to use
    Returns
    -------
    np.ndarray
        The heatmaps as a numpy array
    """

    heatmaps = np.zeros((x_batch.size()[0], x_batch.size()[2], x_batch.size()[3]))
    # iterate over batch and generate heatmaps
    for i in range(x_batch.size()[0]):
        heatmap = explain_image(x_batch[i], model=model, encoder=encoder, layer=layer, embeddings_db=embeddings_db,
                                gradients_db=gradients_db, k=k, plot=plot)
        heatmaps[i] = heatmap

    return heatmaps


def generate_batches(dataset_names, model_names):
    """Generates batches of images from the test set and generates heatmaps for them"""
    for dataset_name in dataset_names:
        # load batch of 64 images from test set
        data_module, reverse_transform = setup_datamodule(dataset_name, batch_size=16)
        data_loader = extract_data_loader(data_module, "test")
        i = 0
        for batch in tqdm(data_loader, desc="Encoding images", leave=False):

            s_batch, x_batch = parse_batch(batch, dataset_name)

            for model_name in model_names:
                work_path, database_path, plot_path, batches_path = setup_paths(method_name, model_name, dataset_name)
                # load model
                model, encoder, layers, transformations = setup_model(model_name)
                layer = layers[0]

                embeddings_db, gradients_db = read_database(database_path)

                # visualize_reconstructions(model, x_batch, save_path=plot_path, reverse_transform=reverse_transform)

                # get heatmaps for batch
                a_batch = explain_batch(model, encoder, layer, embeddings_db, gradients_db, x_batch,
                                        save_path=work_path, plot=False, k=15)

                plot_batches([x_batch, a_batch], is_heatmap=[False, True], n=5,
                             main_title=f"{model_name} for {dataset_name}",
                             plot=False, save_path=plot_path + f"/{i}.png")
                #
                # save all batches
                save_batches(work_path=work_path, x_batch=x_batch, a_batch=a_batch, s_batch=s_batch, iteration=i)
            i += 1


if __name__ == '__main__':
    method_name = "gradcam"
    model_names = ["resnet18", "vae", "simclr"]
    dataset_names = ["two4two"]  # ["cifar10"]["two4two"]


    generate_batches(dataset_names, model_names)
