
from __future__ import unicode_literals, print_function, division

import os.path

import torch, random
from torch.utils.data import Dataset
from loader import SCANDataset, TokenizerSCAN, SCANDataLoader
from encoder_decoder import EncoderGRU, AttentionDecoderGRU, EncoderLSTM, DecoderAttentionLSTM
from bahdanau_trainer import BahDanauTrainer
from torch.utils.data import Subset
import pandas as pd
from pathlib import Path

# Loading the data and training the encoder
DATA_CLOUD = "SCAN/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_result_dir = "exp-3-jump"
# Path(exp_result_dir).mkdir(parents=True, exist_ok=True)
def tokenize_data(data):
    token_data = []
    for i in range(len(data)):
        input_line, output_line = data[i]
        tokenized_input, tokenized_output = tokenizer.encode_inputs(input_line), tokenizer.encode_outputs(output_line)
        token_data.append([tokenized_input, tokenized_output])
    return token_data

# training file: tasks_train_addprim_jump.txt
# testing file: tasks_test_addprim_jump.txt

train_data = SCANDataset(root=DATA_CLOUD, split_type="add_prim_split", target_file='addprim_jump')
test_data = SCANDataset(root=DATA_CLOUD, split_type="add_prim_split",  target_file='addprim_jump', train=False)

tokenizer = TokenizerSCAN()
tokenizer.fit(train_data)

tokenized_train_data = tokenize_data(train_data)
tokenized_test_data = tokenize_data(test_data)

tokenized_train_data[0], tokenized_test_data[0]
print("tokenized_train_data[0]", tokenized_train_data[0])
print("tokenized_test_data[0]", tokenized_test_data[0])

def TEST_TOKENIZER_ON_ORIGINAL_DATA(dataset, tokenized_data):
    for i in range(len(dataset)):
        og_input_data, og_output_data = dataset[i]
        decoded_data_input = tokenizer.decode_inputs(tokenized_data[i][0])
        decoded_data_input = " ".join(decoded_data_input)

        decoded_data_output = tokenizer.decode_outputs(tokenized_data[i][1])
        decoded_data_output = " ".join(decoded_data_output)
        og_input_data = og_input_data + " EOS"
        og_output_data = og_output_data + " EOS"
        if (og_input_data != decoded_data_input):
            print("FAIL: TRAIN DATA")
            print("Expected: ", og_input_data)
            print("Output", decoded_data_input)
            return False

        if (og_output_data != decoded_data_output):
            print("FAIL: OUTPUT DATA")
            print("Expected: ", og_output_data)
            print("Output", decoded_data_output)
            return False
    return True

print("Test the tokenizer on input data")
print(TEST_TOKENIZER_ON_ORIGINAL_DATA(train_data, tokenized_train_data))
print(TEST_TOKENIZER_ON_ORIGINAL_DATA(test_data, tokenized_test_data))

tensor_train_data = SCANDataLoader(train_data, tokenizer)
tensor_test_data = SCANDataLoader(test_data, tokenizer)


# takes 10 samples
indices = torch.arange(10)
tensor_subset_train_data = Subset(tensor_train_data, indices)
tensor_subset_test_data = Subset(tensor_test_data, indices)

print("length of sub sample:", len(tensor_subset_train_data))

# import pdb; pdb.set_trace()

MAX_LENGTH = -1
for inputs, outputs in tokenized_train_data:
    input_size = len(inputs)
    output_size = len(outputs)
    if input_size > MAX_LENGTH:
        MAX_LENGTH = input_size
    if output_size > MAX_LENGTH:
        MAX_LENGTH = output_size
print("max length in the inputs/outputs", MAX_LENGTH)


hidden_size = 100
dropout_prob = 0.1
subset=True

for i in range(5):
    results_dict = {"train": [], "test": []}
    print(f"Seed: {i + 1}")
    seed_val = i+1
    random.seed(i)
    encoder1 = EncoderLSTM(tokenizer.get_n_words(), hidden_size).to(device)
    decoder1 = DecoderAttentionLSTM(hidden_size, tokenizer.get_n_cmds(), dropout_prob, MAX_LENGTH).to(device)
    if subset:
        print("In loop ---")
        BahDanauTrainer.train_loop(encoder1, decoder1, tokenizer, tensor_subset_train_data, 50, exp_result_dir, seed_val, print_every=10, plot_every=5, max_length=MAX_LENGTH)
        results_dict['train'].append(BahDanauTrainer.getDatasetAccuracy(encoder1, decoder1, tokenizer, tensor_subset_train_data))
        results_dict['test'].append(BahDanauTrainer.getDatasetAccuracy(encoder1, decoder1, tokenizer, tensor_subset_test_data))
        BahDanauTrainer.evaluateRandomly(encoder1, decoder1, tokenizer, tensor_subset_test_data, exp_result_dir, mode="test", n=10, plot_attention=True)
    else:
        BahDanauTrainer.train_loop(encoder1, decoder1, tokenizer, tensor_train_data       ,100000, exp_result_dir, seed_val, print_every=2000, plot_every=2000, max_length=MAX_LENGTH)
        results_dict['train'].append(BahDanauTrainer.getDatasetAccuracy(encoder1, decoder1, tokenizer, tensor_train_data))
        results_dict['test'].append(BahDanauTrainer.getDatasetAccuracy(encoder1, decoder1, tokenizer, tensor_test_data))
        BahDanauTrainer.evaluateRandomly(encoder1, decoder1, tokenizer, tensor_test_data, exp_result_dir, mode="test", n=10, plot_attention=True)

    results_pd = pd.DataFrame.from_dict(results_dict) # fix because they aint in diff columns
    # creating a directory where the results will be saved
    results_dir = "results"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_pd.to_csv(os.path.join(results_dir, exp_result_dir + "_seed"+ str(seed_val) + "_" + "results.csv"))

    torch.save(encoder1.state_dict(), os.path.join(exp_result_dir +"_seed"+ str(seed_val) +"_LSTMEncoder.model"))
    torch.save(decoder1.state_dict(), os.path.join(exp_result_dir + "_seed"+ str(seed_val) + "_LSTMDecoder.model"))