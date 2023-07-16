from itertools import product

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from skimage import feature, transform
import torch.nn.functional as F
from torchvision.transforms import transforms

from general_utils import read_batches, setup_paths
from models.bolts import setup_model


def compare_imgs(img1, img2, title_prefix=""):
    # Calculate MSE loss between both images
    loss = F.mse_loss(img1, img2, reduction="sum")
    # Plot images for visual comparison
    grid = torchvision.utils.make_grid(torch.stack([img1, img2], dim=0), nrow=2, normalize=True, range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4, 2))
    plt.title(f"{title_prefix} Loss: {loss.item():4.2f}")
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


def plot_heatmap(heatmap, original, ax, cmap='RdBu_r',
                 percentile=99, dilation=0.5, alpha=0.25):
    """
    Plots the heatmap on top of the original image
    (which is shown by most important edges).

    Parameters
    ----------
    heatmap : Numpy Array of shape [X, X]
        Heatmap to visualise.
    original : Numpy array of shape [X, X, 3]
        Original image for which the heatmap was computed.
    ax : Matplotlib axis
        Axis onto which the heatmap should be plotted.
    cmap : Matplotlib color map
        Color map for the visualisation of the heatmaps (default: RdBu_r)
    percentile : float between 0 and 100 (default: 99)
        Extreme values outside of the percentile range are clipped.
        This avoids that a single outlier dominates the whole heatmap.
    dilation : float
        Resizing of the original image. Influences the edge detector and
        thus the image overlay.
    alpha : float in [0, 1]
        Opacity of the overlay image.

    """
    if len(heatmap.shape) == 3:
        heatmap = np.mean(heatmap, 0)

    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, heatmap.shape[1], dx)
    yy = np.arange(0.0, heatmap.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_original = plt.get_cmap('Greys_r')
    cmap_original.set_bad(alpha=0)
    overlay = None
    if original is not None:
        # Compute edges (to overlay to heatmaps later)
        original_greyscale = original if len(original.shape) == 2 else np.mean(original, axis=-1)
        in_image_upscaled = transform.rescale(original_greyscale, dilation, mode='constant',
                                              multichannel=False, anti_aliasing=True)
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(heatmap), percentile)
    abs_min = abs_max

    ax.imshow(heatmap, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        ax.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_original, alpha=alpha)


def visualize_reconstructions(model, input_imgs, title="Reconstructed", save_path=None, model_name=None,
                              reverse_transform=None):
    """
    Visualize the reconstructions of the given input images.
    Parameters
    ----------
    model : torch.nn.Module
        Model to use for the reconstruction.
    input_imgs : torch.Tensor
        Input images to reconstruct. A batch of images is expected. The
        expected shape is [batch_size, channels, height, width].
    title : str
        Title of the plot.
    save_path : str
        Path to save the plot to.
    model_name : str
        Name of the model. Used for the plot title.
    """
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(input_imgs.to(model.device))
    reconst_imgs = reconst_imgs.cpu()

    # Plotting
    _plot_reconstruction(input_imgs, reconst_imgs, unnormalize=reverse_transform)
    # imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
    # grid = torchvision.utils.make_grid(imgs, nrow=len(input_imgs), normalize=True)
    # grid = grid.permute(1, 2, 0)
    # plt.figure(figsize=(14, 9), dpi=100)
    # plt.title(title + f" ({model_name})" if model_name else title)
    # plt.imshow(grid)
    # plt.axis("off")
    # plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


def _plot_reconstruction(X, X_hat, unnormalize):
    n = min(X.size(0), 8)
    fig, axes = plt.subplots(2, n, figsize=(10, 2))
    for i in range(n):
        ax_real = axes[0][i]
        ax_real.imshow(np.transpose(unnormalize(X[i]), (1, 2, 0)))
        ax_real.get_xaxis().set_visible(False)
        ax_real.get_yaxis().set_visible(False)

        ax_gen = axes[1][i]
        unnormalized_image = unnormalize(X_hat[i])
        # min max normalization
        unnormalized_image = (unnormalized_image - unnormalized_image.min()) / (
                unnormalized_image.max() - unnormalized_image.min())

        unnormalized_image = np.transpose(unnormalized_image.detach().numpy(), (1, 2, 0))

        ax_gen.imshow(unnormalized_image)
        ax_gen.get_xaxis().set_visible(False)
        ax_gen.get_yaxis().set_visible(False)

    plt.tight_layout()

    plt.show()


def plot_img_mask_heatmap(img, mask, heatmap, title="Image with Heatmap", save=False, save_path=None, titles=None):
    img = make_img_plotable(img)
    mask = make_img_plotable(mask)
    if not titles:
        titles = ["Image", "Mask", "Heatmap"]
    fontsize = 24
    images = [img, mask, heatmap]

    f, axarr = plt.subplots(1, len(images), figsize=(16, 8), dpi=100)

    for i, image in enumerate(images):
        hm = axarr[i].imshow(image)
        axarr[i].axis('off')
        axarr[i].set_title(titles[i], fontsize=fontsize)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axarr[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Create colorbar
    cbarlabel = "Importance"
    cbar = axarr[2].figure.colorbar(hm, ax=axarr[2], cax=cax, cmap='RdBu_r')
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=20)

    plt.suptitle(title, fontsize=34, y=0.95)

    if save and save_path is not None:
        plt.savefig(save_path, dpi=300)
        return
    plt.show()

    plt.tight_layout()
    plt.show()


def plot_heatmap_and_img(heatmap, img_tensor, title="Image with Heatmap", save_path=None, plot=True):
    img = make_img_plotable(img_tensor)
    heatmap = make_img_plotable(heatmap)

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
    if plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()


def make_img_plotable(img):
    """
    Transforms a torch tensor into a numpy array that can be plotted with matplotlib.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.shape[0] == 1:
        img = np.squeeze(img, axis=0)
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img -= np.min(img)
    img /= np.max(img)
    return img


# def tmp():
#     def to_np(x):
#         return x.cpu().detach().numpy()
#
#     model_name_list = ['Supervised', 'SimCLR', 'SwAV', 'VAE']
#     relax_list = [ ** explentations
#     for supervised **, ** explentations for SimCLR **, ** explentations for SwAV **, ** explentations for VAE **]
#
#     fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
#
#     font_size = 15
#
#     for idx, (model_explanation, model_name) in enumerate(zip(relax_list, model_name_list)):
#
#         if idx == 0: ax[0][idx + 1].set_ylabel('Importance', fontsize=font_size)
#
#         ax[0][idx + 1].imshow(imsc(x[0]))
#         im = ax[0][idx + 1].imshow(to_np(model_explanation.importance()), alpha=0.75, cmap='bwr')
#         ax[0][idx + 1].set_xticks([])
#         ax[0][idx + 1].set_yticks([])
#         ax[0][idx + 1].set_title(model_name, fontsize=font_size)
#
#         if idx == 0: ax[1][idx + 1].set_ylabel('Uncertainty', fontsize=font_size)
#
#         ax[1][idx + 1].imshow(imsc(x[0]))
#
#         ax[1][idx + 1].imshow(to_np(model_explanation.uncertainty()), alpha=0.75, cmap='bwr')
#         ax[1][idx + 1].set_xticks([])
#         ax[1][idx + 1].set_yticks([])
#         ax[1][idx + 1].set_title(model_name, fontsize=font_size)
#
#     ax[0][0].imshow(imsc(x[0]))
#     ax[0][0].set_xticks([])
#     ax[0][0].set_yticks([])
#     ax[0][0].set_title('Input', fontsize=font_size)
#
#     ax[1][0].imshow(mask.squeeze(), cmap='gist_gray')
#     ax[1][0].set_xticks([])
#     ax[1][0].set_yticks([])
#     ax[1][0].set_title('Ground Truth', fontsize=font_size)
#
#     p0 = ax[1][1].get_position().get_points().flatten()
#     p2 = ax[1][-1].get_position().get_points().flatten()
#     ax_cbar = fig.add_axes([p0[0], 0, p2[2] - p0[0], 0.05])
#     cbar = fig.colorbar(im, orientation="horizontal", cax=ax_cbar)
#     cbar.set_ticks([])
#
#     plt.show()
#
#
# def tmp2():
#     def to_np(x):
#         return x.cpu().detach().numpy()
#
#     model_name_list = ['Supervised', 'SimCLR', 'SwAV', 'VAE']
#
#     fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
#
#     font_size = 15
#
#     for idx, (model_name) in enumerate(model_name_list):
#
#         if idx == 0: ax[0][idx + 1].set_ylabel('G-SHAP', fontsize=font_size)
#
#         ax[0][idx + 1].imshow(imsc(x[0]))
#         im = ax[0][idx + 1].imshow(np.abs(shap_list[idx][0]).sum(
#             axis=0))  # here np.abs(shap_list[idx][0] basically just returns a 3x128x128 importance map (cuz it's RGB)
#         ax[0][idx + 1].set_xticks([])
#         ax[0][idx + 1].set_yticks([])
#         ax[0][idx + 1].set_title(model_name, fontsize=font_size)
#
#         if idx == 0: ax[1][idx + 1].set_ylabel('IG', fontsize=font_size)
#
#         ax[1][idx + 1].imshow(imsc(x[0]))
#
#         ax[1][idx + 1].imshow(np.abs(ig_list[idx][0]).sum(
#             axis=0))  # here np.abs(ig_list[idx][0] basically just returns a 3x128x128 importance map (cuz it's RGB)
#         ax[1][idx + 1].set_xticks([])
#         ax[1][idx + 1].set_yticks([])
#         ax[1][idx + 1].set_title(model_name, fontsize=font_size)
#
#     ax[0][0].imshow(imsc(x[0]))
#     ax[0][0].set_xticks([])
#     ax[0][0].set_yticks([])
#     ax[0][0].set_title('Input', fontsize=font_size)
#
#     ax[1][0].imshow(mask.squeeze(), cmap='gist_gray')  # comment this out if there is no ground truth map
#     ax[1][0].set_xticks([])
#     ax[1][0].set_yticks([])
#     # ax[1][0].set(frame_on=False) #comment this in if there is no ground truth
#     ax[1][0].set_title('Ground Truth', fontsize=font_size)  # comment this out if there is no ground truth map
#
#     p0 = ax[1][1].get_position().get_points().flatten()
#     p2 = ax[1][-1].get_position().get_points().flatten()
#     ax_cbar = fig.add_axes([p0[0], 0, p2[2] - p0[0], 0.05])
#
#     cbar = fig.colorbar(im, orientation="horizontal", cax=ax_cbar)
#     cbar.set_ticks([])
#
#     plt.show()


def convert_batches_to_img_list(batches, n):
    """ by the first dimension (batch size)


    :param batches:
    :param n:
    :return:
    """
    images = []
    for img_index in range(n):
        image_tuple = []
        for batch in batches:
            img = batch[img_index]
            img = make_img_plotable(img)
            image_tuple.append(img)
        images.append(image_tuple)
    return images


def plot_batches(batches, is_heatmap, n=5, main_title="Images and Masks", titles=None, plot = True, save_path = None):
    if titles is None:
        titles = ["Image" if not is_heatmap[i] else "Heatmap" for i in range(len(is_heatmap))]
    # convert batches to images
    images = convert_batches_to_img_list(batches, n)
    fontsize = 14
    matplotlib.rcParams['image.cmap'] = 'bwr'
    figsize = (3 * n, 3 * len(batches))
    # create subplots
    fig, axarr = plt.subplots(nrows=len(batches), ncols=n, figsize=figsize)
    # plot images
    for outer_index, img_tuple in enumerate(images):
        for inner_index, img in enumerate(img_tuple):
            hm = axarr[inner_index, outer_index].imshow(img)
            axarr[inner_index, outer_index].axis('off')
            axarr[inner_index, outer_index].set_title(titles[inner_index], fontsize=fontsize)

            if is_heatmap[inner_index]:
                # Create colorbar
                from mpl_toolkits.axes_grid1 import make_axes_locatable

                divider = make_axes_locatable(axarr[inner_index, outer_index])
                cax = divider.append_axes("right", size="5%", pad=0.05)

                # Create colorbar
                cbarlabel = "Importance"
                cbar = axarr[inner_index, outer_index].figure.colorbar(hm, ax=axarr[inner_index, outer_index], cax=cax,
                                                                       cmap='hot')
                cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=fontsize)
    plt.suptitle(main_title, fontsize=24, y=0.95)
    plt.tight_layout()
    if plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()


if __name__ == '__main__':
    method_name = "gradcam"
    model_names = ["resnet18", "vae", "simclr"]
    dataset_names = ["two4two"]  # ["cifar10"]["two4two"]

    # convert batches to images
    fontsize = 14
    matplotlib.rcParams['image.cmap'] = 'bwr'
    n=2
    figsize = (3 * len(model_names), 3)
    # create subplots
    fig, axarr = plt.subplots(nrows=n, ncols=len(model_names) + 1, figsize=figsize)
    # flatten axarr
    axarr = axarr.flatten()
    titles = ["Image", "ResNet18", "VAE", "SimCLR"]
    images = []
    for i in range (n):
        first_image=True
        for model_name, dataset_name in product(model_names, dataset_names):
            work_path, database_path, plot_path, batches_path = setup_paths(method_name, model_name, dataset_name)
            a_batch, x_batch, s_batch, y_batch = read_batches(batches_path, limit=n)
            if first_image:
                images.append(make_img_plotable(x_batch[i]))
                first_image = False
            images.append(make_img_plotable(a_batch[i]))

    # plot images
    for outer in range(n):
        for inner in range(len(model_names)+ 1):
            index = outer * (len(model_names) + 1) + inner

            hm = axarr[index ].imshow(images[index])
            axarr[index].axis('off')
            axarr[index ].set_title(titles[inner], fontsize=fontsize)

    # add space for colour bar at the bottom
    plt.subplots_adjust(bottom=0.1*2)


    cbar_ax = fig.add_axes([0.325, 0.1, 0.58, 0.06])
    cbar = fig.colorbar(hm, cax=cbar_ax, orientation="horizontal", )

    plt.suptitle("GradCAM", fontsize=18, y=0.95)
    plt.show()

