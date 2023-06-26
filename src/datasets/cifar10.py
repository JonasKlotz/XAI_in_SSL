from pl_bolts.datamodules import CIFAR10DataModule
from torchvision import transforms


def load_cifar10_data_module(data_dir="/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/cifar10", batch_size=32):
    cifar10_dm = CIFAR10DataModule(data_dir=data_dir, batch_size=batch_size)
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        )
    ])
    return cifar10_dm, transformations
