import sys

from xai.embedded_gradcam.helpers import _plot_grad_heatmap, _plot_grad_heatmap_and_img
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

gradients = None
activations = None


def backward_hook(module, grad_input, grad_output):
    global gradients  # refers to the variable in the global scope
    # print('Backward hook running...')
    gradients = grad_output[0]
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



def GradCAM(model,encoder, layer, img_batch, plot=True, img_path=None, model_type:str=None):  # NOSONAR
    """
    Generates a heatmap for the given image tensor
    Args:
        img_batch: the image tensor shape (b, c, h, w)
        model: the model
        layer: the layer to generate the heatmap from
        img_path: the path to the image
        plot: whether to plot the heatmap
        model_type: the model type, vae, simclr or swav
    Returns:

    """
    if img_path is not None:
        image = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        img_tensor = transform(image).unsqueeze(0)
    else:
        img_tensor = img_batch

    heatmap, pooled_gradients, embeddings = _generate_gradcam_heatmap(img_tensor, model, encoder, layer=layer, model_type=model_type)

    if plot:
        _plot_grad_heatmap(heatmap)
        _plot_grad_heatmap_and_img(heatmap, img_tensor)

    return pooled_gradients, embeddings


def _generate_gradcam_heatmap(img_tensor: torch.Tensor, model: torch.nn.Module,encoder: torch.nn.Module, layer: torch.nn.Module, model_type:str='vae'):
    img_tensor = img_tensor.to(device)
    embeddings = encoder(img_tensor)
    if isinstance(embeddings, list):
        embeddings = embeddings[0][0].unsqueeze(0)
    # defines two global scope variables to store our gradients and activations
    global activations
    global gradients

    # register forward hook and backward hook at the layer of interest
    f_hook = layer.register_forward_hook(forward_hook)
    b_hook = layer.register_full_backward_hook(backward_hook)

    loss = get_outputs_loss(img_tensor, model, model_type)
    loss.backward()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
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


def get_outputs_loss(img_tensor, model, model_type):
    """ Returns the loss of the model for the given input tensor """
    if model_type == 'vae':
        batch_tuple = (img_tensor, None)  # tuplerize as models expect that input is a tuple
        outputs = model.step(batch_tuple, batch_idx=0)  # idx doenst matter
        return outputs[0]
    elif model_type == 'simclr':
        batch_tuple = ((img_tensor[0].unsqueeze(0), img_tensor[1].unsqueeze(0), None), None)  # simclr format
        outputs = model.shared_step(batch_tuple)  # idx doenst matter

    elif model_type == 'swav':
        batch_tuple = (img_tensor, None)  # tuplerize as models expect that input is a tuple
        outputs = model.shared_step(batch_tuple)  # idx doenst matter
    elif model_type == 'simclr_trained':
        batch_tuple = ((img_tensor[0].unsqueeze(0), img_tensor[1].unsqueeze(0)), None, None)  # simclr format
        outputs = model.shared_step(batch_tuple)
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return outputs


def generate_activations(model, layer, img_tensor):
    """ Generates the activations for the given image tensor"""
    # defines two global scope variables to store our gradients and activations
    global activations

    # register forward hook and backward hook at the layer of interest
    f_hook = layer.register_forward_hook(forward_hook)

    _ = model(img_tensor.to(device))  # NOSONAR: necessary to get the activations
    tmp_activations = activations.detach().clone()
    activations = None
    f_hook.remove()

    return tmp_activations
