import sys

# Adds the directory to your python path.
sys.path.append("/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/src")

import gc
import os
from typing import List

from PIL import Image
from numpy import matlib as mb
import cv2
import matplotlib as mpl
import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
import skimage.transform as skt
import zarr
from tqdm import tqdm

from datasets.datautils import sample_from_data_module, extract_data_loader, embed_imgs, setup_datamodule
from models.bolts import setup_model
from sklearn.metrics.pairwise import cosine_similarity
from general_utils import save_batches, parse_batch
from visualization.plotting import plot_batches

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


class HookModel(torch.nn.Module):
    def __init__(self, model, layers: List[torch.nn.Module]):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = []

        self.pretrained = model

        for layer in layers:
            h = layer.register_forward_hook(self.forward_hook())
            self.layerhook.append(h)

        for p in self.pretrained.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out.append(out)
            self.tensorhook.append(out.register_hook(self.activations_hook))

        return hook

    def forward(self, x):
        self.selected_out = []
        out = self.pretrained(x)
        return out, self.selected_out

    def clear_hooks(self):
        for hook in self.tensorhook:
            hook.remove()
        self.tensorhook = []
        self.selected_out = []

        for l in self.layerhook:
            l.remove()


def scale_heatmap(size, heatmap, cmap):
    heatmap = cv2.resize(heatmap, size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cmap)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def put_heatmap(img, heatmap, cmap=cv2.COLORMAP_INFERNO, alpha=128):
    heatmap = Image.fromarray(scale_heatmap(img.shape[:2], heatmap, cmap))
    heatmap.putalpha(alpha)
    img = Image.fromarray(img)

    img.paste(heatmap, (0, 0), heatmap)
    return img


def compute_spatial_similarity(conv1, conv2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    pool1 = np.mean(conv1, axis=0)
    pool2 = np.mean(conv2, axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])), int(np.sqrt(conv1.shape[0])))
    conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
    conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    im_similarity = np.zeros((conv1_normed.shape[0], conv1_normed.shape[0]))
    for zz in range(conv1_normed.shape[0]):
        repPx = mb.repmat(conv1_normed[zz, :], conv1_normed.shape[0], 1)
        im_similarity[zz, :] = np.multiply(repPx, conv2_normed).sum(axis=1)
    similarity1 = np.reshape(np.sum(im_similarity, axis=1), out_sz)
    similarity2 = np.reshape(np.sum(im_similarity, axis=0), out_sz)
    return similarity1, similarity2


def preprocess_img(img, transform=None, device=None):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    if transform:
        img = transform(img)
    if device:
        img = img.to(device)

    return img


def get_heatmaps(model, img1, img2, device, transform=None):
    """
    Computes the spatial similarity between the last convolutional layer of two images.
    :param model:
    :param img1:
    :param img2:
    :param device:
    :param transform:
    :return:
    """
    img1 = preprocess_img(img1, transform, device).double()
    img2 = preprocess_img(img2, transform, device).double()
    model = model.to(device)
    model = model.double()

    _, activations1 = model(img1)
    _, activations2 = model(img2)
    heatmaps1 = []
    heatmaps2 = []
    for a1, a2 in zip(activations1, activations2):
        a1 = a1.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        a1 = a1.reshape(-1, a1.shape[-1])
        a2 = a2.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        a2 = a2.reshape(-1, a2.shape[-1])

        h1, h2 = compute_spatial_similarity(a1, a2)
        h1 -= np.min(h1)
        h1 /= (np.max(h1) + np.finfo(float).eps)
        h1 = np.max(h1) - h1

        h2 -= np.min(h2)
        h2 /= (np.max(h2) + np.finfo(float).eps)
        h2 = np.max(h2) - h2

        heatmaps1.append(h1)
        heatmaps2.append(h2)

    return heatmaps1, heatmaps2


def plot_heatmaps(img, img_sim, heatmaps, heatmaps_sim, layer_names):
    f, axarr = plt.subplots(2, len(heatmaps) + 1, figsize=(24, 12))
    fontsize = 24
    axarr[0, 0].imshow(img)
    axarr[0, 0].axis('off')
    axarr[0, 0].set_title(f'Queried Image', fontsize=fontsize)

    axarr[1, 0].imshow(img_sim)
    axarr[1, 0].axis('off')
    axarr[1, 0].set_title(f'Most Similar Image', fontsize=fontsize)

    for i, (h1, h2, l) in enumerate(zip(heatmaps, heatmaps_sim, layer_names)):
        axarr[0, i + 1].imshow(put_heatmap(np.array(img), h1, cmap=cv2.COLORMAP_INFERNO, alpha=180))
        axarr[0, i + 1].axis('off')
        axarr[0, i + 1].set_title(f'Layer {l}', fontsize=fontsize)

        axarr[1, i + 1].imshow(put_heatmap(np.array(img_sim), h2, cmap=cv2.COLORMAP_INFERNO, alpha=180))
        axarr[1, i + 1].axis('off')
        axarr[1, i + 1].set_title(f'Layer {l}', fontsize=fontsize)

    plt.tight_layout()
    plt.colorbar(mpl.cm.ScalarMappable(cmap='inferno'), ax=axarr.ravel().tolist())

    plt.show()


def explain_image(query_img, ds_embeddings, ds_imgs, encoder, layers, device, plot=False):
    encoder = encoder.double()
    query_img = query_img.double()

    # embed query img
    with torch.no_grad():
        query_embeddings = encoder(query_img.to(device))

    if isinstance(query_embeddings, list) or isinstance(query_embeddings, tuple):
        query_embeddings = query_embeddings[0]
    query_embeddings = query_embeddings.cpu().numpy()

    # calc cosine similarity between query imgs and dataset images
    most_similar = cosine_similarity(query_embeddings, ds_embeddings).squeeze(0)
    id_most_similar = np.argsort(-most_similar)[0]
    most_sim_img = ds_imgs[id_most_similar]

    if most_sim_img.shape != query_img.shape[1:]:  # without batch dimension
        most_sim_img = skt.resize(most_sim_img, query_img.shape[1:], anti_aliasing=True)

    # hook model
    hook_model = HookModel(encoder, layers=layers).to(device)

    heatmaps, heatmaps_sim = get_heatmaps(hook_model, query_img, most_sim_img, device, transform=None)

    if plot:
        query_img_pil = post_process_img(query_img)
        most_sim_img_pil = post_process_img(most_sim_img)
        plot_heatmaps(query_img_pil, most_sim_img_pil, heatmaps, heatmaps_sim,
                      layer_names=[str(l) for l in layers] * 2)
    hook_model.clear_hooks()
    return heatmaps, heatmaps_sim


def post_process_img(img_tensor):
    if len(img_tensor.shape) == 4:
        img_tensor = img_tensor.squeeze(0)
    img_tensor -= img_tensor.min()
    img_tensor /= (img_tensor.max() + 0.0000000001)
    if isinstance(img_tensor, np.ndarray):
        img_tensor = torch.from_numpy(img_tensor)
    return transforms.ToPILImage()(img_tensor)


def load_zarr_embeddings(database_path, names=None):
    if names == None:
        names = ['embeddings', 'images']
    results = [zarr.open(os.path.join(database_path, name + '.zarr')) for name in names]
    for i in range(len(results)):
        assert results[i] is not None, f'Could not load {names[i]} from {database_path}'
    return results


def explain_batch(x_batch, encoder, layers, database_path, save_path=None, device=None, plot=False):
    encoder.eval()

    dims = x_batch.shape[-2:]
    a_batch = np.zeros(shape=(len(x_batch), dims[0], dims[1]))
    embeddings_database, images_database, = load_zarr_embeddings(database_path)

    for i in range(len(x_batch)):
        query_img = x_batch[i].unsqueeze(0)
        heatmaps, sim_heatmaps = explain_image(query_img=query_img, ds_embeddings=embeddings_database,
                                               ds_imgs=images_database,
                                               encoder=encoder, layers=layers, device=device, plot=plot)
        heatmap = heatmaps[0]
        heatmap = skt.resize(heatmap, dims)
        a_batch[i] = heatmap

    return a_batch


def generate_databases(encoder, datamodule_name, database_path, device=None, num_batches=10000):
    data_module = setup_datamodule(dataset_name=datamodule_name, batch_size=1)

    data_loader = extract_data_loader(data_module)
    ds_imgs, ds_embeddings = embed_imgs(encoder, data_loader, database_path, device=device, num_batches=num_batches)
    return ds_imgs, ds_embeddings


def create_vsdn_databases(dataset_names, model_names, base_path, n=10000, device=None):
    """
    Creates databases for all models and datasets
    :param dataset_names:
    :param model_names:
    :param base_path:
    :param n:
    :param device:
    :return:
    """
    for dataset_name in dataset_names:
        data_module, _ = setup_datamodule(dataset_name=dataset_name, batch_size=1)
        data_loader = extract_data_loader(data_module)

        for model_name in model_names:
            work_path = os.path.join(base_path, dataset_name, model_name)
            database_path = os.path.join(work_path, "database")
            # load model
            _, encoder, _, _ = setup_model(name=model_name)
            embed_imgs(encoder, data_loader, database_path, device=device, num_batches=n)

            # clear memory
            gc.collect()


def explain_all_batches(dataset_names, model_names, base_path, device=None):
    """  Explain all batches of all models and datasets
    """
    for dataset_name in dataset_names:
        for model_name in model_names:
            work_path = os.path.join(base_path, dataset_name, model_name)
            database_path = os.path.join(work_path, "database")
            plot_path = os.path.join(work_path, "plots")
            os.makedirs(plot_path, exist_ok=True)

            # load model
            model, encoder, layers, _ = setup_model(name=model_name)

            data_module, _ = setup_datamodule(dataset_name=dataset_name, batch_size=64)
            data_loader = extract_data_loader(data_module, stage='test')

            for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Encoding images", leave=False):
                s_batch, x_batch = parse_batch(batch, dataset_name)
                a_batch = explain_batch(x_batch=x_batch, encoder=encoder, layers=layers, database_path=database_path,
                                        save_path=work_path, device=device, plot=False)
                save_batches(work_path, x_batch=x_batch, a_batch=a_batch, s_batch=s_batch, iteration=i)
                plot_batches([x_batch, a_batch], is_heatmap=[False, True], n=5,
                             main_title=f"VDSN: {model_name} for {dataset_name}",
                             plot=False, save_path=plot_path + f"/{i}.png")
            print(f"Finished {model_name} on {dataset_name}")


if __name__ == '__main__':
    base_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results/vsdn"
    model_names = ["swav"]#, "simclr_pretrained", ]  # DONE ["simclr", "vae", "resnet18"]
    dataset_names = ["two4two"]  # DONE ["cifar10", "two4two"]
    n = 10000

    # create_vsdn_databases(dataset_names, model_names, base_path, n=n)

    explain_all_batches(dataset_names, model_names, base_path)
