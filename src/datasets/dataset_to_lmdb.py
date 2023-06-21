import pickle

import lmdb

from datasets.datautils import extract_data_loader
from datasets.two4two import Two4TwoDataModule
from tqdm import tqdm

def dataloader_to_lmdb(data_loader, save_path):
    """
    Converts a data loader to an lmdb database
    """
    l = len(data_loader)
    map_size = 100000000 * l
    env = None
    try:
        env = lmdb.open(save_path, map_size=map_size)

        for i, batch in tqdm(enumerate(data_loader), total=l):
            for subsample_index, img in enumerate(batch):
                key = f"{subsample_index}"
                with env.begin(write=True) as txn:
                    key_bytes = key.encode('ascii')
                    value_bytes = pickle.dumps(batch)
                    txn.put(key_bytes, value_bytes)
    finally:
        env.close()

def try_lmdb(path='data/deepglobe_patches/train'):
    """
    Test the lmdb file by loading it and printing the first subsample.
    """

    # load the lmdb file
    env = lmdb.open(path)
    txn = env.begin()
    # print the length of the lmdb file
    print("Length:", txn.stat()['entries'])
    # get the keys of the lmdb file decoded
    keys = [key.decode('ascii') for key, _ in txn.cursor()]
    print(keys[0])
    # get a single subsample
    values = [txn.get(key.encode()) for key in keys]
    print(values[0])
    values = [pickle.loads(value) for value in values]
    print(values[0])
    labels = [value['labels'] for value in values]
    print(labels[0])



if __name__ == '__main__':
    data_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/two4two"
    work_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results"
    lmdb_path = '/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/data/lmdb'

    data_module = Two4TwoDataModule(data_dir=data_path, working_path=work_path, batch_size=32)
    data_loader = extract_data_loader(data_module)

    #dataloader_to_lmdb(data_loader, lmdb_path)

    try_lmdb(lmdb_path)