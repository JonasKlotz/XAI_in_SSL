import sys

from zarr.errors import ArrayNotFoundError

# Adds the other directory to your python path.
sys.path.append("/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/src")

from models.bolts import setup_model
from xai.embedded_gradcam.gradcam import GradCAM

import numpy as np
from tqdm import tqdm
from datasets.datautils import extract_data_loader, setup_datamodule
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
    try:
        gradients = zarr.open_array(gradients_zarr, mode='r')
        embeddings = zarr.open_array(embeddings_zarr, mode='r')
    except ArrayNotFoundError as e:
        print(e)
        print(f"Failed to read database from database_path: {database_path}")
        raise e
    return embeddings, gradients


def build_all_databases(base_path, model_names, dataset_names, end=5000):

    for dataset_name in dataset_names:

        for model_name in model_names:
            database_path = path.join(base_path, dataset_name, model_name, "database")
            batch_size = 1 if model_name == "vae" else 2

            model, encoder, layers = setup_model(model_name)
            layer = layers[0]

            ###################################################################################################################
            data_module, reverse_transform = setup_datamodule(dataset_name, batch_size=batch_size,
                                                              model_name=model_name)
            try:
                gradients_zarr, embeddings_zarr = build_database(data_module=data_module,
                                                                 model=model,
                                                                 encoder=encoder,
                                                                 database_path=database_path,
                                                                 layer=layer,
                                                                 end=end,
                                                                 model_type=model_name)
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                exit(0)
            except Exception as e:
                print(f"Failed to build database for {model_name} on {dataset_name}")
                print(e)
                continue


if __name__ == "__main__":
    base_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results/gradcam"

    model_names = ["simclr", "vae",]
    dataset_names =  ["cifar10", "two4two"]#["two4two", "cifar10"]
    build_all_databases(base_path, model_names, dataset_names, end=500)
