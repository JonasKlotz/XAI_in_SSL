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

from datasets.cifar10 import load_cifar10_data_module
from datasets.datautils import sample_from_data_module, extract_data_loader, embed_imgs, load_img_to_batch
from datasets.two4two import Two4TwoDataModule
from models.VQVAE import VQVAE
from models.bolts import  load_simclr, load_swav, load_vae
from visualization.plotting import visualize_reconstructions
from xai.metrics.top_k_intersections import TopKIntersection
from sklearn.metrics.pairwise import cosine_similarity

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
    if isinstance(query_embeddings, list):
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
    return [zarr.open(os.path.join(database_path, name + '.zarr')) for name in names]


def explain_batch(x_batch, encoder, layers, database_path, save_path=None, device=None, plot=False):
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

    if save_path:
        # save a_batch
        np.save(save_path + '/batches/' + 'a_batch.npy', a_batch)
    return a_batch


def setup_model(name=None):
    if name == 'simclr':
        model = load_simclr()
        encoder = model.encoder
        layers = [encoder.layer4[1].conv2]  # , encoder.fc]
    elif name == 'vae':
        model = load_vae(input_height=128, input_channels=3)
        encoder = model.encoder
        layers = [encoder.layer4[1].conv1]  # , encoder.fc]
    elif name == 'swav':
        encoder = load_swav()
        layers = [encoder.model.layer4[2].conv3]  # , model.model.projection_head[0]]  # SWAV layer around pooling

    else:
        raise ValueError('name must be either simclair, swav or vae')
    return encoder, layers


def setup_datamodule(dataset_name=None, batch_size=1):
    if dataset_name == 'two4two':
        data_module = Two4TwoDataModule(batch_size=batch_size)
    elif dataset_name == 'cifar10':
        data_module, transforms = load_cifar10_data_module(batch_size=batch_size)
    return data_module


def generate_databases(encoder, datamodule_name, database_path, device=None, num_batches=10000):
    data_module = setup_datamodule(dataset_name=datamodule_name, batch_size=1)

    data_loader = extract_data_loader(data_module)
    ds_imgs, ds_embeddings = embed_imgs(encoder, data_loader, database_path, device=device, num_batches=num_batches)
    return ds_imgs, ds_embeddings


if __name__ == '__main__':
    model_name = 'simclr'
    datamodule_name = 'cifar10'
    work_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results/vsdn"
    database_path = work_path + f"/database/cifar10/{model_name}"

    # load model
    encoder, layers = setup_model(name=model_name)
    # data_module = setup_datamodule(dataset_name=datamodule_name, batch_size=1)
    #
    # data_loader = extract_data_loader(data_module)
    # ds_imgs, ds_embeddings = embed_imgs(encoder, data_loader, database_path, device=device, num_batches=10000)

    data_module = setup_datamodule(dataset_name=datamodule_name, batch_size=4)
    x_batch, _ = sample_from_data_module(data_module, stage='test')
    a_batch = explain_batch(x_batch=x_batch, encoder=encoder, layers=layers, database_path=database_path,
                            save_path=work_path, device=device, plot=True)

    # Metrics
    # #metric = TopKIntersection(k=1000, return_aggregate=True)
    #
    # #batch_eval = metric(x_batch=x_batch, s_batch=s_batch, a_batch=a_batch)
    # #print(batch_eval)
