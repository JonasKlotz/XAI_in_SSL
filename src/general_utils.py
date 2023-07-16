import os
from os import path

import numpy as np
import torch
import zarr

base_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results"


def setup_paths(method_name, model_name, dataset_name):
    method_path = path.join(base_path, method_name)
    work_path = path.join(method_path, dataset_name, model_name)
    database_path = path.join(work_path, "database")
    plot_path = path.join(work_path, "plots")
    batches_path = path.join(work_path, "batches")
    return work_path, database_path, plot_path, batches_path


def save_batches(work_path, x_batch=None, s_batch=None, y_batch=None, a_batch=None, iteration=0):
    remove = iteration == 0
    """Saves the batches as zarr to the work path"""
    if not path.exists(path.join(work_path, 'batches')):
        os.mkdir(path.join(work_path, 'batches'))
    if x_batch is not None:
        _save_batch_as_zarr(work_path, x_batch, 'x_batch', remove=remove)
    if s_batch is not None:
        _save_batch_as_zarr(work_path, s_batch, 's_batch', remove=remove)
    if y_batch is not None:
        _save_batch_as_zarr(work_path, y_batch, 'y_batch', remove=remove)
    if a_batch is not None:
        _save_batch_as_zarr(work_path, a_batch, 'a_batch', remove=remove)


def _save_batch_as_zarr(work_path, batch, name, remove=False):
    """Saves a batch as zarr to the work path"""
    zarr_path = path.join(work_path, 'batches', name + '.zarr')
    if isinstance(batch, torch.Tensor):
        batch = batch.detach().cpu().numpy()

    if remove and path.exists(zarr_path):
        import shutil
        shutil.rmtree(zarr_path)
        print(f"Removed existing zarr at {zarr_path}")
    if path.exists(zarr_path):
        print(f"Appending to existing zarr at {zarr_path}")
        # append to existing zarr
        batch_array = zarr.open(zarr_path, mode='a')
        batch_array.append(batch)
    else:
        print(f"Creating new zarr at {zarr_path}")
        # create new zarr
        batch_array = zarr.open_array(zarr_path, mode='w', shape=batch.shape,
                                      dtype=np.double)
        batch_array[:] = batch


def parse_batch(batch, dataset_name):
    x_batch = batch[0]
    if dataset_name == "two4two":
        s_batch = batch[1]
    else:
        s_batch = None
    return s_batch, x_batch


def read_batches(batches_path, convert_to_numpy=True, limit: int = None):
    """
    Loads the batches from the work path as zarr arrays
    :param limit:
    :param convert_to_numpy:
    :param batches_path:
    :return:
    """
    a_batch = _read_batch(batches_path, "a_batch", convert_to_numpy, limit)
    x_batch = _read_batch(batches_path, "x_batch", convert_to_numpy, limit)
    s_batch = _read_batch(batches_path, "s_batch", convert_to_numpy, limit)
    y_batch = _read_batch(batches_path, "y_batch", convert_to_numpy, limit)
    return a_batch, x_batch, s_batch, y_batch


def _read_batch(batches_path, batch_name, convert_to_numpy, limit):
    try:
        batch = zarr.open(path.join(batches_path, f"{batch_name}.zarr"), mode="r")
        if convert_to_numpy:
            if limit is not None:
                batch = batch[:limit]
            else:
                batch = batch[:]  # getitem returns a numpy array
    except zarr.errors.PathNotFoundError:
        print(f"No {batch_name} found at {batches_path}")
        batch = None
    return batch
