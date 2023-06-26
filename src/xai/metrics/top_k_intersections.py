from typing import Optional, List, Any

import numpy as np
import torch

from xai.metrics.base import Metric


class TopKIntersection(Metric):
    """
    Implementation of the top-k intersection by Theiner et al., 2021. [1]

    Code adapted from Quantis toolkit [2].

    The TopKIntersection implements the pixel-wise intersection between a ground truth target object mask and
    an "explainer" mask, the binarized version of the explanation. High scores are desired, as the
    overlap between the ground truth object mask and the attribution mask should be maximal.

    References:
        1) Jonas Theiner et al.: "Interpretable Semantic Photo
        Geolocalization." arXiv preprint arXiv:2104.14995 (2021).
        2) Hedström, Anna, et al. "Quantus: an explainable AI toolkit for responsible evaluation of neural network
        explanations." arXiv preprint arXiv:2202.06861 (2022).
    """

    def __init__(
            self,
            k: int = 1000,
            concept_influence: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        # Save metric-specific attributes.
        self.k = k
        self.concept_influence = concept_influence

    def __call__(
            self,
            x_batch: np.array,
            model=None,
            y_batch: Optional[np.ndarray] = None,
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            device: Optional[str] = None,
            batch_size: int = 64,
            **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).
        """
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            device=device,
            **kwargs,
        )

    def evaluate_instance(
            self,
            model: torch.nn.Module,
            x: np.ndarray,
            y: np.ndarray,
            a: np.ndarray,
            s: np.ndarray,
            device=None,
            **kwargs,
    ):
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        Parameters
        ----------
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x: np.ndarray
            The input to be evaluated on an instance-basis.
        y: np.ndarray
            The output to be evaluated on an instance-basis.
        a: np.ndarray
            The explanation to be evaluated on an instance-basis.
        s: np.ndarray
            The segmentation to be evaluated on an instance-basis.

        Returns
        -------
        float
            The evaluation results.
            :param device:
        """

        # Prepare shapes.
        s = s.astype(bool)
        top_k_binary_mask = np.zeros(a.shape)

        # Sort and create masks.
        sorted_indices = np.argsort(a, axis=None)
        np.put_along_axis(top_k_binary_mask, sorted_indices[-self.k:], 1, axis=None)
        top_k_binary_mask = top_k_binary_mask.astype(bool)

        # Top-k intersection.
        tki = 1.0 / self.k * np.sum(np.logical_and(s, top_k_binary_mask))

        # Concept influence (with size of object normalised tki score).
        if self.concept_influence:
            tki = np.prod(s.shape) / np.sum(s) * tki

        return tki
