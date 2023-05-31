import PIL
import numpy as np
import torch.nn.functional as F
from PIL import Image
from matplotlib import colormaps, pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from datasets.datautils import extract_data_loader

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

    img_tensor.to(device)
    heatmap, pooled_gradients, embeddings = _generate_gradcam_heatmap(img_tensor, model)

    if plot:
        _plot_grad_heatmap(heatmap, img_tensor)

    return pooled_gradients, embeddings


def _plot_grad_heatmap(heatmap, img_tensor):
    # draw the heatmap
    plt.matshow(heatmap.detach())
    # Show the plot
    plt.show()
    # Create a figure and plot the first image
    fig, ax = plt.subplots()
    ax.axis('off')  # removes the axis markers
    # First plot the original image
    ax.imshow(to_pil_image(img_tensor, mode='RGB'))
    # Resize the heatmap to the same size as the input image and defines
    # a resample algorithm for increasing image resolution
    # we need heatmap.detach() because it can't be converted to numpy array while
    # requiring gradients
    overlay = to_pil_image(heatmap.detach(), mode='F').resize((256, 256), resample=PIL.Image.BICUBIC)
    # Apply any colormap you want
    cmap = colormaps['jet']
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Plot the heatmap on the same axes,
    # but with alpha < 1 (this defines the transparency of the heatmap)
    ax.imshow(overlay, alpha=0.4, interpolation='nearest', extent='extent')
    # Show the plot
    plt.show()


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
    heatmap /= torch.max(heatmap)

    # cleanup
    b_hook.remove()
    f_hook.remove()
    gradients = None
    activations = None

    return heatmap, pooled_gradients, embeddings


def collect_embeddings_and_gradients(model, data_loader, pickle_path):
    # Encode all images in the data_loader using model, and return both images and encodings
    grad_dict = {}
    model.eval()  # not nograd
    i = 0
    for imgs, _ in tqdm(data_loader, desc="Encoding images", leave=False):
        pooled_gradients, embeddings = GradCAM(model, imgs, plot=False)
        # convert to numpy array
        pooled_gradients = pooled_gradients.detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()

        # convert to float 16 to save memory
        pooled_gradients = pooled_gradients.astype(np.float16)
        embeddings = embeddings.astype(np.float16)
        # add pooled_gradients and embeddings to array
        # embed_array, grad_array = _combine_arrays(embed_array, embeddings, grad_array, pooled_gradients)
        grad_dict[embeddings.tobytes()] = pooled_gradients
        i += 1
        if i % 200 == 0:
            grad_dict = _dump_dictionary(grad_dict, pickle_path)

    _dump_dictionary(grad_dict, pickle_path)


def _pickle_array(embed_array, grad_array, pickle_path):
    with open(pickle_path, 'ab') as f:
        pickle.dump((embed_array, grad_array), f)

    # reset arrays to save memory
    del grad_array, embed_array
    grad_array = None
    embed_array = None
    return embed_array, grad_array


def _dump_dictionary(grad_dict, pickle_path):
    """Dumps a dictionary to a pickle file"""
    # if file exists, append to it
    if path.exists(pickle_path):
        with open(pickle_path, 'ab') as f:
            pickle.dump(grad_dict, f)
    else:
        with open(pickle_path, 'wb') as f:
            pickle.dump(grad_dict, f)
    del grad_dict
    return {}


def build_database(data_path, work_path, model_path):
    """
    Builds a database of embeddings and gradients for all images in the data_path's training loader
    :param data_path:
    :param work_path:
    :param model_path:
    :return:
    """
    pck_path = path.join(work_path, "grad_array.pkl")
    data_module = Two4TwoDataModule(data_dir=data_path, working_path=work_path)
    model = VQVAE.load_from_checkpoint(model_path, map_location=device)
    # img_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/test.png"
    # GradCAM(model, img_path)

    data_loader = extract_data_loader(data_module, "fit")
    collect_embeddings_and_gradients(model, data_loader, pck_path)
    #  array is saved in a format of [embedding, gradient]
    print("done")


if __name__ == '__main__':
    data_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/two4two"
    work_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results"
    model_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results/VAE.ckpt"

    build_database(data_path, work_path, model_path)
