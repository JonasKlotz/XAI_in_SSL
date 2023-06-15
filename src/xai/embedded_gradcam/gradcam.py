import sys

from xai.embedded_gradcam.helpers import _plot_grad_heatmap, _plot_grad_heatmap_and_img
from PIL import Image
from torchvision import transforms
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

gradients = None
activations = None


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


def GradCAM(model, img_batch, layer_name, plot=True, img_path=None):  # NOSONAR

    if img_path is not None:
        image = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        img_tensor = transform(image).unsqueeze(0)
    else:
        img_tensor = img_batch

    heatmap, pooled_gradients, embeddings = _generate_gradcam_heatmap(img_tensor, model, layer_name=layer_name)

    if plot:
        _plot_grad_heatmap(heatmap)
        _plot_grad_heatmap_and_img(heatmap, img_tensor)

    return pooled_gradients, embeddings


def _generate_gradcam_heatmap(img_tensor, model, layer_name):
    # defines two global scope variables to store our gradients and activations
    global activations
    global gradients

    # register forward hook and backward hook at the layer of interest
    f_hook = model.attr(layer_name).register_forward_hook(forward_hook)
    b_hook = model.attr(layer_name).register_backward_hook(backward_hook)

    # todo: different model outputs??
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
