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