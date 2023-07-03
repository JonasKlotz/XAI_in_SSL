import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


def load_vae(input_height: int = 32, pretrained: bool = False, from_pretrained:str = 'cifar10-resnet18',
               **kwargs: object) -> "L.LightningModule":
    """
    Load a VAE model from PyTorch Lightning Bolts

    :param pretrained:
    :param kwargs:input_height: int = 32,
    :return:
    """
    from pl_bolts.models.autoencoders import VAE
    vae = VAE(input_height, **kwargs)
    print(vae.pretrained_urls)
    if pretrained:
        vae = vae.from_pretrained(from_pretrained)
    return vae


def load_simclr():
    from pl_bolts.models.self_supervised import SimCLR
    simclr_weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    simclr = SimCLR.load_from_checkpoint(simclr_weight_path, strict=False, map_location=device)
    simclr = simclr.to(device)
    simclr.eval()
    return simclr


def load_swav():
    from pl_bolts.models.self_supervised.swav.swav_module import SwAV
    swav_weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
    swav = SwAV.load_from_checkpoint(swav_weight_path, strict=False)
    swav.dataset = "none" # doesnt matter
    swav = swav.to(device)
    swav.eval()
    return swav

def setup_model(name=None, input_height=32):
    if name == 'simclr':
        model = load_simclr()
        encoder = model.encoder
        layers = [encoder.layer2[2].conv3] # [encoder.layer1[2].conv2] # [encoder.layer1[2].conv3] torch.Size([1, 256, 8, 8]) #[encoder.layer2[2].conv3] torch.Size([1, 512, 4, 4]) #[encoder.layer2[2].conv2] [1, 128, 4, 4])  # , encoder.fc]
    elif name == 'vae':
        model = load_vae(input_height=input_height, pretrained=True)
        encoder = model.encoder
        layers = [encoder.layer4[1].conv1]  # , encoder.fc]
    elif name == 'swav':
        model = load_swav()

        encoder  = model.model
        layers = [encoder.layer4[2].conv2]  # , model.model.projection_head[0]]  # SWAV layer around pooling

    else:
        raise ValueError('name must be either simclair, swav or vae')
    return model, encoder, layers



if __name__ == '__main__':
    # simclr = load_simclr()
    # last_conv_layer  = simclr.encoder.layer4[-1].conv3

    vqvae = load_vae()
    print(vqvae)


