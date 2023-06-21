from abc import abstractmethod
from typing import Optional, Dict, Callable, Union, Any, Collection

import numpy as np
import torch


class Metric:
    """
    Implementation of the base Metric class.
    """

    def __init__(
            self,
            return_aggregate: bool = False,
            aggregate_func: Callable= np.mean,
    ):

        # Save metric-specific attributes.
        self.return_aggregate = return_aggregate
        self.aggregate_func = aggregate_func

        # Save metric-specific attributes.
        self.last_results = None

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Optional[np.ndarray],
            a_batch: Optional[np.ndarray],
            s_batch: Optional[np.ndarray],
            device: Optional[str] = None,
            batch_size: int = 64,

            **kwargs,
    ) -> Union[int, float, list, dict, Collection[Any], None]:
        """
                This implementation represents the main logic of the metric and makes the class object callable.
                It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
                output labels (y_batch) and a torch model (model).

                Parameters
                ----------
                model: torch.nn.Module
                x_batch: np.ndarray
                    A np.ndarray which contains the input data that are explained.
                y_batch: np.ndarray
                    A np.ndarray which contains the output labels that are explained.
                a_batch: np.ndarray, optional
                    A np.ndarray which contains pre-computed attributions i.e., explanations.
                s_batch: np.ndarray, optional
                    A np.ndarray which contains segmentation masks that matches the input.
                device: string
                    Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
                kwargs: optional
                    Keyword arguments.

                Returns
                -------
                last_results: list
                    a list of Any with the evaluation scores of the concerned batch.
                """
        self.last_results = [None for _ in x_batch]
        for i in range(x_batch.shape[0]):
            x = self.tensor_to_numpy(x_batch[i])
            y = self.tensor_to_numpy(y_batch[i]) if y_batch is not None else None
            a = self.tensor_to_numpy(a_batch[i]) if a_batch is not None else None
            s = self.tensor_to_numpy(s_batch[i]) if s_batch is not None else None

            result = self.evaluate_instance(
                model=model,
                x=x,
                y=y,
                a=a,
                s=s,
                device=device,
                **kwargs,
            )
            self.last_results[i] = result

        if self.return_aggregate:
            if self.aggregate_func:
                try:
                    self.last_results = [self.aggregate_func(self.last_results)]
                except:
                    print(
                        "The aggregation of evaluation scores failed. Check that "
                        "'aggregate_func' supplied is appropriate for the data "
                        "in 'last_results'."
                    )
            else:
                raise KeyError(
                    "Specify an 'aggregate_func' (Callable) to aggregate evaluation scores."
                )

        return self.last_results

    @abstractmethod
    def evaluate_instance(
            self,
            model: torch.nn.Module,
            x: np.ndarray,
            y: Optional[np.ndarray],
            a: Optional[np.ndarray],
            s: Optional[np.ndarray],
    ) -> Any:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        This method needs to be implemented to use __call__().

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
        Any
        """
        raise NotImplementedError()

    def tensor_to_numpy(self, tensor) -> np.ndarray:
        """
        Convert the tensor to numpy array.

        Returns
        -------
        np.ndarray
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        else:
            return np.array(tensor)
