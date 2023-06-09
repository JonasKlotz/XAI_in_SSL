import tarfile
import os
import os.path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def extract(tar_file, path):
    """Extracts the tar file to the path specified
    Args:
        tar_file: the tar file
        path: the path to extract the tar file to
    """

    if not os.path.isfile(tar_file) :
        print("The not a file")
        return

    opened_tar = tarfile.open(tar_file)

    if tarfile.is_tarfile(tar_file):
        opened_tar.extractall(path)
    else:
        print("The tar file you entered is not a tar file")


def extract_data_loader(data_module, stage="fit"):
    """Extracts the data loader from the data module
    Args:
        data_module: the data module
        stage: the stage of the data loader
    Returns:
        the data loader
        """
    data_module.prepare_data()
    data_module.setup(stage)
    if stage == "fit":
        data_loader = data_module.train_dataloader()
    elif stage == "test":
        data_loader = data_module.test_dataloader()
    elif stage == "predict":
        data_loader = data_module.predict_dataloader()
    return data_loader


def load_img_to_batch(img_path):
    # generate embeddings from image
    img = Image.open(img_path)
    # convert img to rgb
    img = img.convert('RGB')
    # convert to tensor
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    return img_tensor


def sample_from_data_module(data_module, stage="fit"):
    """Samples a batch from the data module
    Args:
        data_module: the data module
        stage: the stage of the data loader
    Returns:
        the batch
    """
    data_loader = extract_data_loader(data_module, stage)
    batch = next(iter(data_loader))
    return batch


def embed_imgs(encoder, data_loader, num_batches=100, device = None):
    # Encode all images in the data_loader using model, and return both images and encodings
    img_list, embed_list = [], []
    encoder.eval()
    if device:
        encoder.to(device)
    i = 0
    for batch in tqdm(data_loader, desc="Encoding images", leave=False):
        imgs = batch[0]
        if device:
            imgs = imgs.to(device)

        with torch.no_grad():
            embeddings = encoder(imgs)
        img_list.append(imgs)
        embed_list.append(embeddings)
        i += 1
        if num_batches and i == num_batches:
            break
    return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0))
