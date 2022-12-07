from __future__ import unicode_literals, print_function, division
import torch
from reduced_training import get_reduced_training_set
from training import trainIters
from encoder_rnn import EncoderRNN
from attn_decoder import AttnDecoderRNN, AttnDecoderBahdanau
from evaluation import evaluate, evaluateRandomly
import numpy as np
from utils import get_tokenizer, MAX_LENGTH, tokenizer, tokenize_data, \
    hidden_size, dropout_prob, device, get_torched_vec_data, train_data, test_data


# from process_data import tokenize_data
def get_tokenizer():
    return TokenizerSCAN()

tokenizer = get_tokenizer()

tokenizer.fit(train_data)
import pdb; pdb.set_trace()


# Get tokenized version of the data
tokenized_train_data = tokenize_data(train_data)
tokenized_test_data = tokenize_data(test_data)

print("Check the tokenized train and test samples")
print(tokenized_train_data[0], tokenized_test_data[0])


# Test if the encoder/decoder is correct
def TEST_TOKENIZER_ON_ORIGINAL_DATA(dataset, tokenized_data):
    for i in range(len(dataset)):
        og_input_data, og_output_data = dataset[i]
        decoded_data_input = tokenizer.decode_inputs(tokenized_data[i][0])
        decoded_data_input = " ".join(decoded_data_input)

        decoded_data_output = tokenizer.decode_outputs(tokenized_data[i][1])
        decoded_data_output = " ".join(decoded_data_output)
        og_input_data = og_input_data + " EOS SOS"
        og_output_data = og_output_data + " EOS"
        if og_input_data != decoded_data_input:
            print("FAIL: TRAIN DATA")
            print("Expected: ", og_input_data)
            print("Output", decoded_data_input)
            return False

        if og_output_data != decoded_data_output:
            print("FAIL: OUTPUT DATA")
            print("Expected: ", og_output_data)
            print("Output", decoded_data_output)
            return False
    return True


# Verify if the encoder/decoder is working as is
TEST_TOKENIZER_ON_ORIGINAL_DATA(train_data, tokenized_train_data)
TEST_TOKENIZER_ON_ORIGINAL_DATA(test_data, tokenized_test_data)

# Find the max length of the tokenizers
for inputs, outputs in tokenized_train_data:
    input_size = len(inputs)
    output_size = len(outputs)
    if input_size > MAX_LENGTH:
        MAX_LENGTH = input_size
    if output_size > MAX_LENGTH:
        MAX_LENGTH = output_size
print(MAX_LENGTH)

torch_train = get_torched_vec_data(tokenized_train_data)
torch_validate = get_torched_vec_data(tokenized_train_data)

# TODO: Somewhere here to get the unique values in the list


percents_to_try = [0.01, 0.02, 0.04, 0.08, 0.16]
results_dict = {percent: {"train": [], "test": []} for percent in percents_to_try}


def getDatasetAccuracy(encoder, decoder, test_data):
    correct_array = np.zeros(len(test_data), dtype=bool)
    for i in range(len(test_data)):
        output_words, attentions = evaluate(encoder, decoder, test_data[i][0])
        y = tokenizer.decode_outputs(test_data[i][1].flatten().detach().to("cpu").numpy())
        y_pred = output_words
        correct_array[i] = y == y_pred
    return np.array(correct_array).sum() / len(correct_array)


for percent in percents_to_try:
    for i in range(3):
        print(f"Percent {percent} iter: {i + 1}")
        # TO DO: Add value to text
        pairs = get_reduced_training_set(torch_train, percent)
        encoder1 = EncoderRNN(tokenizer.get_n_words(), hidden_size).to(device)
        attn_decoder1 = AttnDecoderBahdanau(hidden_size, tokenizer.get_n_cmds(), dropout_p=dropout_prob).to(device)
        trainIters(encoder1, attn_decoder1, 10000, print_every=1000, plot_every=1000)
        results_dict[percent]["train"].append(getDatasetAccuracy(encoder1, attn_decoder1, pairs))
        results_dict[percent]["test"].append(getDatasetAccuracy(encoder1, attn_decoder1, torch_validate))

print("Train results:", results_dict[percent]["train"])

# evaluateRandomly(encoder1, attn_decoder1, pairs)
