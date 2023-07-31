import os

import torch
from lightly.transforms import SimCLRTransform
from lightning import LightningModule
from torch import nn

from models.SimCLR import SimCLRModel

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


def load_vae(input_height: int = 32, pretrained: bool = False, from_pretrained: str = 'cifar10-resnet18',
             **kwargs: object) -> "L.LightningModule":
    """
    Load a VAE model from PyTorch Lightning Bolts

    :param pretrained:
    :param kwargs:input_height: int = 32,
    :return:
    """
    from pl_bolts.models.autoencoders import VAE
    vae = VAE(input_height, **kwargs)
    if pretrained:
        vae = vae.from_pretrained(from_pretrained)
    return vae


def load_simclr():
    from pl_bolts.models.self_supervised import SimCLR
    simclr_weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    simclr = SimCLR.load_from_checkpoint(simclr_weight_path, strict=False, map_location=device)
    simclr = simclr.to(device)
    return simclr


def load_simclr_pretrained(path):
    # model = SimCLRModel().load_from_checkpoint(path, strict=False, map_location=device)
    import __main__
    setattr(__main__, "SimCLRModel", SimCLRModel)
    model = torch.load(path, map_location=device)
    return model


def load_swav():
    from pl_bolts.models.self_supervised.swav.swav_module import SwAV
    swav_weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
    swav = SwAV.load_from_checkpoint(swav_weight_path, strict=False, map_location=torch.device('cpu'))
    swav.dataset = "none"  # doesnt matter
    swav = swav.to(device)
    swav.eval()
    return swav


def setup_model(model_name,
                dataset_name,
                input_height=32,
                model_path="/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results/models/two4two/full_model3.pth"):

    num_classes = 10 if dataset_name == 'cifar10' else 2
    input_height = 32 if dataset_name == 'cifar10' else 128

    if model_name == 'simclr':
        model = load_simclr()
        encoder = model.encoder
        layers = [encoder.layer2[2].conv3]
        transform = SimCLRTransform(
            input_size=input_height, vf_prob=0.5, rr_prob=0.5, cj_prob=0.0, random_gray_scale=0.0
        )
    elif model_name == 'vae':
        model = load_vae(input_height=input_height, pretrained=True)
        encoder = model.encoder
        layers = [encoder.layer4[1].conv1]  # , encoder.fc]
        transform = lambda x: x
    elif model_name == 'swav':
        model = load_swav()
        encoder = model.model
        layers = [encoder.layer2[2].conv2]  # , 4x4[encoder.layer2[2].conv2]  # , 4x4
        transform = lambda x: x
    elif model_name == 'simclr_pretrained':
        model = load_simclr_pretrained(model_path)
        encoder = model
        layers = [model.backbone[5][1].conv2]
        transform = SimCLRTransform(
            input_size=input_height, vf_prob=0.5, rr_prob=0.5, cj_prob=0.0, random_gray_scale=0.0
        )
    elif model_name == 'resnet18':
        model = ImagenetResnet(num_target_classes=num_classes)
        encoder = model
        layers = [model.feature_extractor[4][1].conv2]
        transform = lambda x: x

    else:
        raise ValueError('name must be either simclair, swav or vae')
    return model, encoder, layers, transform


import torchvision.models as models


class ImagenetResnet(LightningModule):
    def __init__(self, num_target_classes: int = 10):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet18(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        self.classifier = nn.Linear(num_filters, num_target_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        return representations

    def shared_step(self, batch):
        x, y = batch
        representations = self(x)
        logits = self.classifier(representations)
        prediction = self.softmax(logits)
        loss = nn.CrossEntropyLoss()(logits, y)
        return loss, prediction, y


# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.
import torch.distributed as dist
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



if __name__ == '__main__':
    model = ImagenetResnet()
    print("DONE")
