from PIL import Image
from numpy import matlib as mb
import cv2
import matplotlib as mpl
import torch
import numpy as np
from matplotlib import pyplot as plt

from datasets.datautils import sample_from_data_module
from datasets.two4two import Two4TwoDataModule
from models.VQVAE import VQVAE


class HookModel(torch.nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = []

        self.pretrained = model

        for i in layers:
            self.pretrained.convs[i].register_forward_hook(self.forward_hook())

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


def get_heatmaps(model, img1, img2, device, transform):
    _, activations1 = model(transform(img1).unsqueeze(0).to(device))
    _, activations2 = model(transform(img2).unsqueeze(0).to(device))
    heatmaps1 = []
    heatmaps2 = []
    print(len(activations1), activations1[0].shape)
    for a1, a2 in zip(activations1, activations2):
        a1 = a1.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        a1 = a1.reshape(-1, a1.shape[-1])
        a2 = a2.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        a2 = a2.reshape(-1, a2.shape[-1])

        print(a1.shape, a2.shape)

        h1, h2 = compute_spatial_similarity(a1, a2)
        h1 -= np.min(h1)
        h1 /= np.max(h1)
        h1 = np.max(h1) - h1

        h2 -= np.min(h2)
        h2 /= np.max(h2)
        h2 = np.max(h2) - h2

        heatmaps1.append(h1)
        heatmaps2.append(h2)

    return heatmaps1, heatmaps2


def plot_heatmaps(img, img_sim, heatmaps, heatmaps_sim, layers):
    f, axarr = plt.subplots(2, len(heatmaps) + 1, figsize=(24, 12))
    axarr[0, 0].imshow(np.array(img))
    axarr[0, 0].axis('off')
    axarr[1, 0].imshow(img_sim)
    axarr[1, 0].axis('off')

    for i, (h1, h2, l) in enumerate(zip(heatmaps, heatmaps_sim, layers)):
        axarr[0, i + 1].imshow(put_heatmap(np.array(img), h1, cmap=cv2.COLORMAP_INFERNO, alpha=180))
        axarr[0, i + 1].axis('off')

        axarr[1, i + 1].set_title(f'Layer {l}', pad=90, size=34)
        axarr[1, i + 1].imshow(put_heatmap(np.array(img_sim), h2, cmap=cv2.COLORMAP_INFERNO, alpha=180))
        axarr[1, i + 1].axis('off')
    plt.tight_layout()
    plt.colorbar(mpl.cm.ScalarMappable(cmap='inferno'), ax=axarr.ravel().tolist(), shrink=0.8)


if __name__ == '__main__':
    data_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/two4two"
    work_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results"
    model_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results/VAE.ckpt"
    database_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results/database"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # img_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/test.png"
    # img_tensor = load_img_to_batch(img_path)
    model = VQVAE.load_from_checkpoint(model_path, map_location=device)
    data_module = Two4TwoDataModule(data_dir=data_path, working_path=work_path, batch_size=4)

    sample, _ = sample_from_data_module(data_module)

    # plot real images
    plt.figure(figsize=(10, 4))
    from torchvision.utils import make_grid

    plt.axis('off')
    plt.imshow(make_grid(sample, normalize=True, value_range=(-1, 1), nrow=len(sample)).permute(1, 2, 0))