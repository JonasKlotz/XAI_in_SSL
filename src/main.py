from datasets.two4two import Two4TwoDataModule
from models.VQVAE import VQVAE
from models.training_utils import train_vqvae_datamodule

if __name__ == '__main__':
    data_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/two4two"
    work_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results"

    data_module = Two4TwoDataModule(data_dir=data_path, working_path=work_path)

    model, results = train_vqvae_datamodule(working_path=work_path, datamodule=data_module, max_epochs=1)