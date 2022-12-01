import numpy as np


def get_reduced_training_set(data_to_sample, percentage, total_n=100000):
    subset_n = int(len(data_to_sample) * percentage)
    subset_i = np.random.choice(np.arange(len(data_to_sample)), subset_n)
    subsampled_i = np.random.choice(subset_i, total_n)
    subsampled_data = [data_to_sample[i] for i in subsampled_i]
    return subsampled_data


def get_unique_inputs():
    """This returns all the unique inputs from the file"""

percents_to_try = [0.01, 0.02, 0.04, 0.08, 0.16]
results_dict = {percent:{"train":[], "test":[]} for percent in percents_to_try}

pairs = get_reduced_training_set(torch_train, 0.01)


# for percent in percents_to_try:
#   for i in range(3):
#     print(f"Percent {percent} iter: {i+1}")
#     pairs = get_reduced_training_set(torch_train, percent)
#     encoder1 = EncoderRNN(tokenizer.get_n_words(), hidden_size).to(device)
#     attn_decoder1 = AttnDecoderRNN(hidden_size, tokenizer.get_n_cmds(), dropout_p=dropout_prob).to(device)
#     trainIters(encoder1, attn_decoder1, 10000, print_every=1000, plot_every=1000)
#     results_dict[percent]["train"].append(getDatasetAccuracy(encoder1, attn_decoder1, pairs))
#     results_dict[percent]["test"].append(getDatasetAccuracy(encoder1, attn_decoder1, torch_validate))