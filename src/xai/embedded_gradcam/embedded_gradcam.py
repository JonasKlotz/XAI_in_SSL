import sys

from xai.embedded_gradcam.database import build_database
from xai.embedded_gradcam.gradcam import GradCAM, forward_hook

print(sys.path)
sys.path.append("/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/src")

#pio.renderers.default = 'png'

import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from datasets.datautils import extract_data_loader
import plotly.express as px
import zarr
from datasets.two4two import Two4TwoDataModule
from models.VQVAE import VQVAE
import torch
from os import path
import pickle

gradients = None
activations = None

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
    # remove the batch dimension
    #    embeddings = embeddings.squeeze()
    # calculate the distance between the embeddings and the embeddings in the database
    distances = np.linalg.norm(embeddings_db - embeddings, axis=0)
    # sum the distances across the last 2 channels
    distances = np.sum(distances, axis=(1, 2))

    # get the indices of the k smallest distances
    indices = np.argpartition(distances, k)[:k]

    return indices


if __name__ == '__main__':
    data_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/two4two"
    work_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results"
    model_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results/VAE.ckpt"
    database_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results/database"

    img_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/test.png"

    model = VQVAE.load_from_checkpoint(model_path, map_location=device)
    data_module = Two4TwoDataModule(data_dir=data_path, working_path=work_path)

    GradCAM(model, img_batch=None, plot=True, img_path=img_path)

    build_database(data_module, work_path, model, database_path, n=100)

    # embeddings_db, gradients_db = read_database(database_path)
    #
    # # generate embeddings from image
    # img = Image.open(img_path)
    # # convert img to rgb
    # img = img.convert('RGB')
    # # convert to tensor
    # img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    #
    # # register forward hook
    # f_hook = model._pre_vq_conv.register_forward_hook(forward_hook)
    # loss, reconstruction, perplexity, embeddings = model(img_tensor.to(device))
    #
    # # get gradients index from database via a k nearest neighbour search
    # k = 1
    # # get k nearest neighbours gradients
    # index = get_k_nearest_neighbours(embeddings, embeddings_db, k)
    #
    # # get the gradients from the database
    # gradients = gradients_db[index]
    #
    # print("done")
