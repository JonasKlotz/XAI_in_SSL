import torch
from pl_bolts.datamodules import CIFAR10DataModule
from torchvision import transforms as transforms_lib
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


def load_cifar10_data_module(data_dir="/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/cifar10", batch_size=32, resize=None):
    cf10_transforms = [transforms_lib.ToTensor(), cifar10_normalization()]
    if resize:
        cf10_transforms += [transforms_lib.Resize(resize, antialias=True)]
    cf10_transforms = transforms_lib.Compose(cf10_transforms)

    cifar10_dm = CIFAR10DataModule(data_dir=data_dir, batch_size=batch_size, normalize=True)

    mean = torch.tensor(cifar10_dm.default_transforms().transforms[1].mean)
    std = torch.tensor(cifar10_dm.default_transforms().transforms[1].std)
    unnormalize = transforms_lib.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return cifar10_dm, unnormalize
