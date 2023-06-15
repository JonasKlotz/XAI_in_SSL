import numpy as np
import plotly.express as px

gradients = None
activations = None


def _plot_grad_heatmap_and_img(heatmap, img_tensor):
    heatmap = heatmap.detach().numpy()
    img = img_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
    # normalize the tensor
    img = img - np.min(img)
    img = img / np.max(img)
    fig = px.imshow(img)
    fig.show()
    #
    # # normalize the heatmap
    # heatmap = heatmap - np.min(heatmap)
    # heatmap = heatmap / np.max(heatmap)
    #
    #
    # # todo make work
    # from plotly.subplots import make_subplots
    # import plotly.graph_objects as go
    # fig = make_subplots(rows=1, cols=2,
    #                     horizontal_spacing=0.01,
    #                     shared_yaxes=True)
    #
    # fig.add_trace(go.Image(z=img), row=1, col=1)
    # fig.add_trace(go.Image(z=heatmap), row=1, col=2)
    # fig.show()

def _plot_grad_heatmap(heatmap):
    data = heatmap.detach()
    fig = px.imshow(data)
    fig.show()
