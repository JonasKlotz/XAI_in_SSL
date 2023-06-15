import torch


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


def load_vqvae(input_height=32, **kwargs):
    from pl_bolts.models.autoencoders import VAE
    vae = VAE(input_height=32, **kwargs)
    print(VAE.pretrained_weights_available())
    vae = vae.from_pretrained('cifar10-resnet18')

    return vae


def load_simclr():
    from pl_bolts.models.self_supervised import SimCLR
    simclr_weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    simclr = SimCLR.load_from_checkpoint(simclr_weight_path, strict=False)
    simclr = simclr.to(device)
    simclr.eval()
    return simclr


def load_swav():
    from pl_bolts.models.self_supervised.swav.swav_module import SwAV
    swav_weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
    swav = SwAV.load_from_checkpoint(swav_weight_path, strict=False)
    swav = swav.to(device)
    swav.eval()
    return swav


if __name__ == '__main__':
    #simclr = load_simclr()
    #last_conv_layer  = simclr.encoder.layer4[-1].conv3


    vqvae = load_vqvae()
    print(vqvae)


