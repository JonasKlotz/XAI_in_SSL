import torch

def get_black_baseline(
    explicand: torch.Tensor
) -> torch.Tensor:
    """
    Get black baseline values for a batch of explicand inputs.

    Args:
    ----
        explicand: A batch of explicands with shape `(batch_size, *)`, where `*`
            indicates the model input size for one sample.
        dataset_name: Name of the dataset where the explicands come from. This is to
            determine the channel means and standard deviations for normalization.
        normalize: Whether to normalize the black baseline values.

    Returns
    -------
        Black baseline values with shape `(batch_size, *)`, on the same device as
        `explicand`.
    """
    black_baseline = torch.zeros_like(explicand)
    return black_baseline
