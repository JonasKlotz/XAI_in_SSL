import sys

print(sys.path)
sys.path.append("/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/src")
import plotly.io as pio

#pio.renderers.default = 'png'

import PIL
import numpy as np
import torch.nn.functional as F
from PIL import Image
from matplotlib import colormaps, pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from datasets.datautils import extract_data_loader
import plotly.express as px
import zarr
from datasets.two4two import Two4TwoDataModule
from models.VQVAE import VQVAE
import torch
from os import path
import pickle
from memory_profiler import profile

gradients = None
activations = None

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


def backward_hook(module, grad_input, grad_output):
    global gradients  # refers to the variable in the global scope
    # print('Backward hook running...')
    gradients = grad_output
    # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
    # print(f'Gradients size: {gradients[0].size()}')
    # We need the 0 index because the tensor containing the gradients comes
    # inside a one element tuple.


def forward_hook(module, args, output):
    global activations  # refers to the variable in the global scope
    # print('Forward hook running...')
    activations = output
    # In this case, we expect it to be a torch.Size([batch size, 1024, 8, 8])
    # print(f'Activations size: {activations.size()}')


def GradCAM(model, img_batch, plot=True, img_path=None):  # NOSONAR

    if img_path is not None:
        image = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        img_tensor = transform(image).unsqueeze(0)
    else:
        img_tensor = img_batch

    heatmap, pooled_gradients, embeddings = _generate_gradcam_heatmap(img_tensor, model)

    if plot:
        _plot_grad_heatmap(heatmap)
        _plot_grad_heatmap_and_img(heatmap, img_tensor)

    return pooled_gradients, embeddings

def _plot_grad_heatmap_and_img(heatmap, img_tensor):
    heatmap = heatmap.detach().numpy()
    img = img_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
    # normalize the tensor
    img = img - np.min(img)
    img = img / np.max(img)
    fig = px.imshow(img)
    fig.show()
    #
    # # normalize the heatmap
    # heatmap = heatmap - np.min(heatmap)
    # heatmap = heatmap / np.max(heatmap)
    #
    #
    # # todo make work
    # from plotly.subplots import make_subplots
    # import plotly.graph_objects as go
    # fig = make_subplots(rows=1, cols=2,
    #                     horizontal_spacing=0.01,
    #                     shared_yaxes=True)
    #
    # fig.add_trace(go.Image(z=img), row=1, col=1)
    # fig.add_trace(go.Image(z=heatmap), row=1, col=2)
    # fig.show()

def _plot_grad_heatmap(heatmap):
    data = heatmap.detach()
    fig = px.imshow(data)
    fig.show()



def _generate_gradcam_heatmap(img_tensor, model):
    # defines two global scope variables to store our gradients and activations
    global activations
    global gradients

    f_hook = model._pre_vq_conv.register_forward_hook(forward_hook)
    b_hook = model._pre_vq_conv.register_full_backward_hook(backward_hook)

    loss, reconstructed, perplexity, embeddings = model(img_tensor.to(device))  # [0].backward()

    loss.backward()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    # weight the channels by corresponding gradients
    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    heatmap = F.relu(heatmap)
    # normalize the heatmap
    heatmap -= torch.min(heatmap)
    heatmap /= torch.max(heatmap)

    # cleanup
    b_hook.remove()
    f_hook.remove()
    gradients = None
    activations = None

    return heatmap, pooled_gradients, embeddings



def collect_embeddings_and_gradients(model, data_loader, gradients_zarr_path, embeddings_zarr_path, end=1000):
    # Encode all images in the data_loader using model, and return both images and encodings
    embeddings_zarr = None
    gradients_zarr = None

    model.eval()  # not nograd
    i = 0
    for imgs, _ in tqdm(data_loader, desc="Encoding images", leave=False):
        if i == end:
            break
        i += 1

        pooled_gradients, embeddings = GradCAM(model, imgs, plot=False)
        # convert to numpy array
        pooled_gradients = pooled_gradients.detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()

        # convert to float 16 to save memory
        pooled_gradients = pooled_gradients.astype(np.float16)
        embeddings = embeddings.astype(np.float16)

        # add batch dimension
        pooled_gradients = np.expand_dims(pooled_gradients, axis=0)
        embeddings = np.expand_dims(embeddings, axis=0)

        if not embeddings_zarr:
            # initialize zarr arrays
            gradients_zarr = zarr.array(pooled_gradients)
            embeddings_zarr = zarr.array(embeddings)
        else:  # store in zarr
            gradients_zarr.append(pooled_gradients, axis=0)
            embeddings_zarr.append(embeddings, axis=0)

    zarr.save(gradients_zarr_path, gradients_zarr)
    zarr.save(embeddings_zarr_path, embeddings_zarr)


def _dump_dictionary(grad_dict, pickle_path):
    """Dumps a dictionary to a pickle file"""
    # if file exists, append to it
    if path.exists(pickle_path):
        with open(pickle_path, 'ab') as f:
            pickle.dump(grad_dict, f)
    else:
        with open(pickle_path, 'wb') as f:
            pickle.dump(grad_dict, f)
    del grad_dict  # free memory
    return {}


def build_database(data_path, work_path, model, database_path, n=30):
    """
    Builds a database of embeddings and gradients for all images in the data_path's training loader
    :param data_path:
    :param work_path:
    :param model_path:
    :return:
    """
    gradients_zarr = path.join(database_path, "grad_array.zarr")
    embeddings_zarr = path.join(database_path, "embs_array.zarr")
    data_module = Two4TwoDataModule(data_dir=data_path, working_path=work_path)
    # img_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/test.png"
    # GradCAM(model, img_path)

    data_loader = extract_data_loader(data_module, "fit")
    collect_embeddings_and_gradients(model, data_loader, gradients_zarr, embeddings_zarr, end=n)
    #  array is saved in a format of [embedding, gradient]
    print("done")


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

    GradCAM(model, img_batch=None, plot=True, img_path=img_path)

    build_database(data_path, work_path, model, database_path, n=100)

    embeddings_db, gradients_db = read_database(database_path)

    # generate embeddings from image
    img = Image.open(img_path)
    # convert img to rgb
    img = img.convert('RGB')
    # convert to tensor
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)

    # register forward hook
    f_hook = model._pre_vq_conv.register_forward_hook(forward_hook)
    loss, reconstruction, perplexity, embeddings = model(img_tensor.to(device))

    # get gradients index from database via a k nearest neighbour search
    k = 1
    # get k nearest neighbours gradients
    index = get_k_nearest_neighbours(embeddings, embeddings_db, k)

    # get the gradients from the database
    gradients = gradients_db[index]

    print("done")
