import numpy as np
import plotly.express as px
import torch
import matplotlib.pyplot as plt

from visualization.vsdn import put_heatmap


def _plot_grad_heatmap_and_img(heatmap, img_tensor, title="Image mask and heatmap", titles=None,
                               save_path=None):
    if isinstance(img_tensor, torch.Tensor):
        img_tensor -= img_tensor.min()
        img_tensor /= img_tensor.max()
        img = img_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()

    f, axarr = plt.subplots(1, 2, figsize=(16, 8), dpi=200)
    axarr[0].imshow(img)
    axarr[0].axis('off')
    axarr[0].set_title("Image", fontsize=24)


    hm = axarr[1].imshow(heatmap)
    axarr[1].set_title("Heatmap", fontsize=24)
    axarr[1].axis('off')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axarr[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Create colorbar
    cbarlabel = "Importance"
    cbar = axarr[1].figure.colorbar(hm, ax=axarr[1], cax=cax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=20)

    plt.suptitle(title, fontsize=34, y=0.95)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        return
    plt.show()


def _plot_grad_heatmap_and_img(heatmap, img_tensor, title="Image with Heatmap", save=False, save_path=None):
    if isinstance(img_tensor, torch.Tensor):
        img_tensor -= img_tensor.min()
        img_tensor /= img_tensor.max()
        img = img_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()

    f, axarr = plt.subplots(1, 2, figsize=(16, 8), dpi=200)
    axarr[0].imshow(img)
    axarr[0].axis('off')
    axarr[0].set_title("Image", fontsize=24)

    hm = axarr[1].imshow(heatmap)
    axarr[1].set_title("Heatmap", fontsize=24)
    axarr[1].axis('off')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axarr[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Create colorbar
    cbarlabel = "Importance"
    cbar = axarr[1].figure.colorbar(hm, ax=axarr[1], cax=cax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=20)

    plt.suptitle(title, fontsize=34, y=0.95)

    if save and save_path is not None:
        plt.savefig(save_path, dpi=300)
        return
    plt.show()


def _plot_grad_heatmap(heatmap, image_tensor=None, mask=None, title="", titles=None, save_path=None):
    if image_tensor is not None:
        _plot_grad_heatmap_and_img(heatmap, image_tensor, title=title, save_path=save_path)
        return
    fig, ax = plt.subplots()
    # Plot the heatmap
    im = ax.imshow(heatmap)
    cbarlabel = "Importance"
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_title(title)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        return
    plt.show()
