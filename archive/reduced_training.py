import numpy as np


def get_reduced_training_set(data_to_sample, percentage, total_n=100000):
    subset_n = int(len(data_to_sample) * percentage)
    subset_i = np.random.choice(np.arange(len(data_to_sample)), subset_n)
    subsampled_i = np.random.choice(subset_i, total_n)
    subsampled_data = [data_to_sample[i] for i in subsampled_i]
    return subsampled_data


def get_unique_inputs():
    """This returns all the unique inputs from the file"""
    # TODO


