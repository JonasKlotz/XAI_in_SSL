from os import path

base_path = "/home/jonasklotz/Studys/23SOSE/XAI_in_SSL/results"

def setup_paths(method_name, model_name, dataset_name):
    method_path = path.join(base_path, method_name)
    work_path = path.join(method_path, dataset_name, model_name)
    database_path = path.join(work_path, "database")
    plot_path = path.join(work_path, "plots")
    batches_path = path.join(work_path, "batches")
    return work_path, database_path, plot_path, batches_path