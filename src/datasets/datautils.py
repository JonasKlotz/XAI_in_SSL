import tarfile
import os
import os.path
# simple function to extract the train data
# tar_file : the path to the .tar file
# path : the path where it will be extracted
def extract(tar_file, path):
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