import torch
import torch.nn as nn
import torch.nn.functional as F

class RELAX(nn.Module):
    """
    RELAX: RunnEstimate Learning via Adaptive eXploration
    A PyTorch implementation of the RELAX algorithm for importance and uncertainty estimation.

    Parameters:
        x (torch.Tensor): The input image to be explained.
        f (torch.nn.Module): The encoder network being used.
        num_batches (int): The number of batches to draw 'batch_size' masks from.
        batch_size (int): Number of masks to draw per batch for feature importance estimation.

    Attributes:
        device (str): The device on which the tensor computations are performed (e.g., 'cuda' or 'cpu').
        shape (tuple): The shape of the input images, excluding batch size and channels.
        pdist (torch.nn.CosineSimilarity): The comparison metric used to compute cosine similarity.
        encoder (torch.nn.Module): The encoder function f used to create embeddings of the input tensor.
        h_star (torch.Tensor): The embedding of the unmasked input image, expanded to match the batch size.
        R (torch.Tensor): Importance scores initialized as zeros.
        U (torch.Tensor): Uncertainty scores initialized as zeros.
        sum_of_weights (torch.Tensor): A small value needed for running mean initialization.

    Methods:
        forward(): Performs the full forward pass using RELAX to generate feature importances and uncertainty scores.
        importance(): Return the estimated feature importance scores.
        uncertainty(): Return the estimated uncertainty scores.
        mask_generator(num_cells=7, p=0.5, nsd=2): Generator function to generate masks for importance estimation.
    """
    def __init__(self, x, f, num_batches, batch_size):
        super().__init__()

        self.device = x.device
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.shape = tuple(x.shape[2:])
        self.pdist = nn.CosineSimilarity(dim=1) 

        self.x = x
        self.encoder = f
        self.h_star = f(x).expand(batch_size, -1)

        self.R = torch.zeros(self.shape, device=self.device)
        self.U = torch.zeros(self.shape, device=self.device)

        self.sum_of_weights = (1e-10)*torch.ones(self.shape, device=self.device)

    def forward(self):
	"""
        Perform the forward pass through the RELAX algorithm to estimate importance and uncertainty.

        Returns:
            None: This method does not return any value.
        """
        for batch in range(self.num_batches):
            for masks in self.mask_generator():

                x_mask = self.x * masks
                h = self.encoder(x_mask)
                sims = self.pdist(self.h_star, h)

                for si, mi in zip(sims, masks.squeeze()):
                    #one-pass implementation of RELAX
                    #i.e. we're using estimators of running mean and variance
                    W_prev = self.sum_of_weights
                    self.sum_of_weights += mi

                    R_prev = self.R.clone()
                    self.R = self.R + mi*(si-self.R) / self.sum_of_weights
                    self.U = self.U + (si-self.R) * (si-R_prev) * mi

        return None

    def importance(self):
        """
        Get the estimated importance scores.

        Returns:
            torch.Tensor: The estimated importance scores.
        """
        return self.R

    def uncertainty(self):
        """
        Get the estimated uncertainty scores.

        Returns:
            torch.Tensor: The estimated uncertainty scores.
        """
        return self.U / (self.sum_of_weights - 1)

    def mask_generator(self, num_cells=7, p=0.5, nsd=2):
        """
        Generator function to generate masks for importance estimation.

        Parameters:
            num_cells (int): Number of cells to break the image into for mask generation.
            p (float): Probability of a cell being masked.
            nsd (int): Number of spatial dimensions (2 for images).

        Yields:
            torch.Tensor: The generated masks.
        """
        #breaks up image into num_cells*num_cells patches, which are randomly occluded in relax
        pad_size = (num_cells // 2, num_cells // 2, num_cells // 2, num_cells // 2)
        grid = (torch.rand(self.batch_size, 1, *((num_cells,) * nsd), device=self.device) < p).float()

        #upsample mask to image size
        grid_up = F.interpolate(grid, size=(self.shape), mode='bilinear', align_corners=False)
        grid_up = F.pad(grid_up, pad_size, mode='reflect')

        #generate random ints to additionally randomly shift a mask by some pixels
        shift_x = torch.randint(0, num_cells, (self.batch_size,), device=self.device)
        shift_y = torch.randint(0, num_cells, (self.batch_size,), device=self.device)

        masks = torch.empty((self.batch_size, 1, self.shape[-2], self.shape[-1]), device=self.device)

        for bi in range(self.batch_size):
            #upscale randomly shifted masks
            masks[bi] = grid_up[bi, :,
                                shift_x[bi]:shift_x[bi] + self.shape[-2],
                                shift_y[bi]:shift_y[bi] + self.shape[-1]]

        yield masks

