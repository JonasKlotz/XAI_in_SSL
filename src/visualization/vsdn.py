from typing import List

from PIL import Image
from numpy import matlib as mb
import cv2
import matplotlib as mpl
import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from datasets.datautils import sample_from_data_module, extract_data_loader, embed_imgs, load_img_to_batch
from datasets.two4two import Two4TwoDataModule
from models.VQVAE import VQVAE
from models.bolts import load_vqvae


class HookModel(torch.nn.Module):
    def __init__(self, model, layers:List[torch.nn.Module]):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = []

        self.pretrained = model

        for layer in layers:
            layer.register_forward_hook(self.forward_hook())

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


def get_heatmaps(model, img1, img2, device, transform=None):
    if transform:
        img1 = transform(img1)
        img2 = transform(img2)

    _, activations1 = model(img1.to(device))
    _, activations2 = model(img2.to(device))
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
    plt.show()


if __name__ == '__main__':
    data_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/two4two"
    work_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results"
    model_path = '/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results/models/bolt_vae.ckpt'
    database_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results/database"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    img_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/test.png"
    img_tensor = load_img_to_batch(img_path)
    # model = VQVAE.load_from_checkpoint(model_path, map_location=device)
    model = load_vqvae(input_height=128, input_width=128, input_channels=3)
    model = model.load_from_checkpoint(model_path, map_location=device)


    encoder = model.encoder

    data_module = Two4TwoDataModule(data_dir=data_path, working_path=work_path, batch_size=4)
    transformations = data_module.transform

    query_imgs, _, _ = sample_from_data_module(data_module, stage='test')
    query_img = query_imgs[0]
    # expand dims
    query_img = query_img.unsqueeze(0)
    # # plot real images
    # plt.figure(figsize=(10, 4))
    from torchvision.utils import make_grid

    #
    # plt.axis('off')
    # plt.imshow(make_grid(query_imgs, normalize=True, nrow=len(query_imgs)).permute(1, 2, 0))
    # plt.show()
    data_loader = extract_data_loader(data_module)
    ds_imgs, ds_embeddings = embed_imgs(encoder, data_loader, device=device, num_batches=15)
    print(ds_imgs.shape, ds_embeddings.shape)

    # embed query imgs
    with torch.no_grad():
        query_embeddings = encoder(query_img.to(device)).cpu().numpy()

    from sklearn.metrics.pairwise import cosine_similarity

    # calc cosine similarity between query imgs and dataset images
    most_similar = cosine_similarity(query_embeddings, ds_embeddings)
    id_most_similar = np.argsort(-most_similar, axis=1)[:, 0]
    most_sim_img = ds_imgs[id_most_similar]

    # plt.figure(figsize=(10, 5))
    # plt.axis('off')
    # plt.title('Query', fontsize=20, loc='left')
    # plt.title('Most similar', fontsize=20, loc='right')
    # plt.imshow(
    #     make_grid(torch.cat([query_img, most_sim_img]), normalize=True, nrow=2).permute(1, 2, 0))
    # plt.show()

    # hook model
    layers = [encoder.layer4[1].conv2]  # last conv layer?
    hook_model = HookModel(encoder, layers=layers).to(device)

    # # denormalize tensors and convert to PIL images
    # query_img_pil = transforms.ToPILImage()((query_img.squeeze(0) + 1) / 2)
    # most_sim_img_pil = transforms.ToPILImage()((most_sim_img.squeeze(0) + 1) / 2)

    heatmaps, heatmaps_sim = get_heatmaps(hook_model, query_img, most_sim_img, device, transform=None)
    # remove batch dim for images
    query_img = query_img.squeeze(0)
    query_img -= query_img.min()
    query_img /= query_img.max()

    most_sim_img = most_sim_img.squeeze(0)
    most_sim_img -= most_sim_img.min()
    most_sim_img /= most_sim_img.max()



    query_img_pil = transforms.ToPILImage()(query_img)
    most_sim_img_pil = transforms.ToPILImage()(most_sim_img)
    plot_heatmaps(query_img_pil, most_sim_img_pil, heatmaps, heatmaps_sim, layers)




