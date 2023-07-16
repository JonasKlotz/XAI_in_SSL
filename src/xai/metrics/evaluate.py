import sys
# Adds the directory to your python path.
sys.path.append("/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/src")

import os

import numpy as np
import pandas as pd
from itertools import product

from general_utils import read_batches
import quantus


def convert_segmentation_map_to_binary(s_batch):
    # sum over all channels
    s_batch = np.sum(s_batch, axis=1)
    s_batch = np.expand_dims(s_batch, axis=1)  # only first channel
    s_batch[s_batch > 0] = 1
    return s_batch


if __name__ == '__main__':
    base_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results"
    method_name = ["vsdn", "gradcam"]
    model_names = ["simclr", "vae", "resnet18"]
    dataset_names =  ["two4two"]

    metric_names = ["top_k_intersection", "pointing_game", "relevance_rank_accuracy"]
    metrics = [quantus.TopKIntersection(k=1000, return_aggregate=True),
               quantus.PointingGame(return_aggregate=True),
               quantus.RelevanceRankAccuracy(return_aggregate=True)]

    df = pd.DataFrame(columns=["method_name", "model_name", "dataset_name", "metric_name", "metric_value"])
    for method_name, model_name, dataset_name in product(method_name, model_names,dataset_names):
        work_path = os.path.join(base_path, method_name, dataset_name, model_name)



        batches_path = os.path.join(work_path, "batches")
        a_batch, x_batch, s_batch, y_batch = read_batches(batches_path)
        # convert segmentation map to binary
        s_batch = convert_segmentation_map_to_binary(s_batch)

        for metric_name, metric in zip(metric_names, metrics):
            print(f"Calculating {metric_name} for {method_name} on {model_name} on {dataset_name}")
            batch_eval = metric(model=None, y_batch=None, x_batch=x_batch, s_batch=s_batch, a_batch=a_batch)
            batch_eval = round(batch_eval[0], 4)

            # fill dataframe
            new_row = {"method_name": method_name, "model_name": model_name, "dataset_name": dataset_name,
                       "metric_name": metric_name, "metric_value": batch_eval}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    # safe dataframe as csv
    df.to_csv(os.path.join(base_path, "metrics.csv"), index=False)
