import matplotlib.pyplot as plt
import torch
import torchvision


from umap import umap_ as ump

from datasets.cifar10 import load_cifar10_data_module
from datasets.datautils import embed_imgs, sample_from_data_module
from models.bolts import load_simclr, load_vqvae
from visualization.plotting import visualize_reconstructions

def find_similar_images(query_img, query_z, key_embeds, K=8):
    # Find closest K images. We use the euclidean distance here but other like cosine distance can also be used.
    dist = torch.cdist(query_z[None, :], key_embeds[1], p=2)
    dist = dist.squeeze(dim=0)
    dist, indices = torch.sort(dist)
    # Plot K closest images
    imgs_to_display = torch.cat([query_img[None], key_embeds[0][indices[:K]]], dim=0)
    grid = torchvision.utils.make_grid(imgs_to_display, nrow=K + 1, normalize=True, range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(12, 3))
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


def plot_similar_images(model, train_loader, test_loader):
    train_img_embeds = embed_imgs(model, train_loader)  # memory problems
    test_img_embeds = embed_imgs(model, test_loader)

    # Plot the closest images for the first N test images as example
    for i in range(8):
        find_similar_images(test_img_embeds[0][i], test_img_embeds[1][i], key_embeds=train_img_embeds)
        # fixme


def plot_umap(layer: torch.nn.Module):
    """ Plots the UMAP projection of the embedding space of the VQ-VAE.

    Args:
        model: Trained VQ-VAE model

        """
    proj = ump.UMAP(n_neighbors=3,
                     min_dist=0.1,
                     metric='cosine').fit_transform(layer.weight.data.cpu())

    plt.scatter(proj[:, 0], proj[:, 1], alpha=0.3)
    plt.imsave('umap.png', proj)
    plt.show()



if __name__ == '__main__':

    # Load data
    cifar_data_module, transforms = load_cifar10_data_module()
    x_batch, y_batch = sample_from_data_module(cifar_data_module, stage='fit')

    # Load model
    model = load_vqvae(pretrained=True)
    encoding_layer = model.encoder.layer4[1].conv2
    #visualize_reconstructions(model, x_batch[:4])
    plot_umap(encoding_layer)
