import sys

from xai.embedded_gradcam.gradcam import GradCAM

import numpy as np
from tqdm import tqdm
from datasets.datautils import extract_data_loader
import zarr
from os import path


def build_database(data_module, model, encoder, database_path, layer, end=None, model_type=None):
    """
    Builds a database of embeddings and gradients for all images in the data_path's training loader
    """
    gradients_zarr = path.join(database_path, "grad_array.zarr")
    embeddings_zarr = path.join(database_path, "embs_array.zarr")

    data_loader = extract_data_loader(data_module, "fit")
    gradients_zarr, embeddings_zarr = collect_embeddings_and_gradients(model, encoder, data_loader, gradients_zarr,
                                                                       embeddings_zarr, layer=layer, end=end,
                                                                       model_type=model_type)
    #  array is saved in a format of [embedding, gradient]
    print(f"Saved database to {database_path}")
    return gradients_zarr, embeddings_zarr


def collect_embeddings_and_gradients(model, encoder, data_loader, gradients_zarr_path, embeddings_zarr_path,
                                     layer, end=None, model_type=None):
    # Encode all images in the data_loader using model, and return both images and encodings
    embeddings_zarr = None
    gradients_zarr = None

    model.eval()  # not nograd
    i = 0
    for batch in tqdm(data_loader, desc="Encoding images", leave=False):
        imgs = batch[0]
        if end and i == end:
            break
        i += 1

        pooled_gradients, embeddings = GradCAM(model, encoder, layer, imgs, plot=False, model_type=model_type)
        # convert to numpy array
        pooled_gradients = pooled_gradients.detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()

        # # convert to float 16 to save memory
        # pooled_gradients = pooled_gradients.astype(np.float16)
        # embeddings = embeddings.astype(np.float16)

        # add batch dimension
        pooled_gradients = np.expand_dims(pooled_gradients, axis=0)
        # embeddings = np.expand_dims(embeddings, axis=0)

        if not embeddings_zarr and not gradients_zarr:
            # initialize zarr arrays
            chunk_size = (50, *pooled_gradients.shape[1:])
            gradients_zarr = zarr.open_array(gradients_zarr_path, mode='w', shape=pooled_gradients.shape,
                                             dtype=np.double, chunks=chunk_size)
            gradients_zarr[0] = pooled_gradients[0]

            chunk_size = (50, *embeddings.shape[1:])
            embeddings_zarr = zarr.open_array(embeddings_zarr_path, mode='w', shape=embeddings.shape, dtype=np.double,
                                              chunks=chunk_size)
            embeddings_zarr[0] = embeddings[0]
        else:  # store in zarr
            gradients_zarr.append(pooled_gradients, axis=0)
            embeddings_zarr.append(embeddings, axis=0)

    # zarr.save(gradients_zarr_path, gradients_zarr)
    # zarr.save(embeddings_zarr_path, embeddings_zarr)
    return gradients_zarr, embeddings_zarr


def read_database(database_path):
    gradients_zarr = path.join(database_path, "grad_array.zarr")
    embeddings_zarr = path.join(database_path, "embs_array.zarr")

    gradients = zarr.open_array(gradients_zarr, mode='r')
    embeddings = zarr.open_array(embeddings_zarr, mode='r')
    return embeddings, gradients
