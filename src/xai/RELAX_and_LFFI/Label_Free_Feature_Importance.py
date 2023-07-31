import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, GradientShap

class AuxiliaryModule(Module):
    """
    An auxiliary module that serves as a black-box to apply classic explanation methods.
    Implements the Label-Free Feature Importance scheme, 
    i.e. it "adds" a inner product of the output embedding with itself to the end of the computational graph.

    Parameters:
        black_box (torch.nn.Module): The PyTorch Module whose output embedding we would like to explain.
        base_features (torch.Tensor): A tensor containing the input features we would like to explain.

    Attributes:
        black_box (torch.nn.Module): The provided black-box module for prediction.
        base_features (torch.Tensor): The tensor containing the input features.
        prediction (torch.Tensor): The prediction made by the black-box module using base_features.

    Methods:
        forward(input_features): Forward pass through the auxiliary module.

    Note:
        The black_box must be a PyTorch Module that takes base_features as input and returns predictions.
        The input_features passed to the forward() method are the features whose importance is to be estimated.
    """
    def __init__(self, black_box, base_features):
	super().__init__()
	self.black_box = black_box
	self.base_features = base_features
	self.prediction = black_box(base_features)

    def forward(self, input_features):
        """
        Forward pass through the auxiliary module.

        Parameters:
            input_features (torch.Tensor): Features whose importance is to be estimated.

        Returns:
            torch.Tensor: Output scalar, final node of computational graph.
        """
	#if only one input
	if len(self.prediction) == len(input_features):
	    #dot product
	    return torch.sum(self.prediction * self.black_box(input_features), dim=-1)
	#if we have an input batch
	elif len(input_features) % len(self.prediction) == 0:
	    n_repeat = int(len(input_features) / len(self.prediction))
	    #dot product
	    return torch.sum(
		self.prediction.repeat(n_repeat, 1) * self.black_box(input_features),
		dim=-1,
	    )

def lffi(encoder, data_loader, device, attr_method, baseline):
"""
    Label-free feature importance wrapper for estimating feature importance.

    Parameters:
        encoder (torch.nn.Module): The encoder model to be used for feature extraction.
        data_loader (torch.utils.data.DataLoader): The data loader for the input data to be explained.
        device (str): The device on which the computations are performed (e.g., 'cuda' or 'cpu').
        attr_method: An attribution method from Captum (e.g., GradientShap or IntegratedGradients).
        baseline: The baseline for the attribution method, usually completly black.

    Returns:
        np.ndarray: Concatenated feature importance scores estimated using the given attribution method.

    Example:
        # Example usage to estimate feature importance using LFFI
        encoder_model = MyEncoder()  # Replace MyEncoder with your own Torch encoder model
        data_loader = torch.utils.data.DataLoader(...)  # Replace ... with your data loading scheme
        attribution_method = captum.attr.GradientShap(encoder_model)  # Choose any attribution method
        baseline = torch.zeros(1, input_size)  # Replace input_size with the appropriate size (c, w, h)
        importance_scores = lffi(encoder_model, data_loader, 'cuda', attribution_method, baseline)
        plt.imshow(np.abs(importance_scores[0].flatten())  # Display the feature importance scores for 1st image
    """
    attributions = []
    for inputs, _ in data_loader:
	inputs = inputs.to(device)
	auxiliary_encoder = AuxiliaryModule(encoder, inputs)
	attr_method.forward_func = auxiliary_encoder
	attributions.append(
	    attr_method.attribute(inputs, baseline).detach().cpu().numpy()
	)
    return np.concatenate(attributions) 
