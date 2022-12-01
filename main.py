from dataloader import SCANDatasetLoader, TokenizerSCAN
from test_tokenizer import TEST_TOKENIZER_ON_ORIGINAL_DATA
from __future__ import unicode_literals, print_function, division
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_CLOUD = "SCAN"
hidden_size = 200 # As per the paper
dropout_prob = 0.5 # As per the paper

train_data = SCANDatasetLoader(root=DATA_CLOUD, split_type="simple_split")
test_data = SCANDatasetLoader(root=DATA_CLOUD, split_type="simple_split", train=False)


def tokenize_data(data):
    token_data = []
    for i in range(len(data)):
        input_line, output_line = data[i]
        tokenized_input, tokenized_output = tokenizer.encode_inputs(input_line), tokenizer.encode_outputs(output_line)
        token_data.append([tokenized_input, tokenized_output])
    return token_data


tokenizer = TokenizerSCAN()
tokenizer.fit(train_data)

# Get tokenized version of the data
tokenized_train_data = tokenize_data(train_data)
tokenized_test_data = tokenize_data(test_data)

print("Check the tokenized train and test samples")
print(tokenized_train_data[0], tokenized_test_data[0])

# Verify if the encoder/decoder is working as is
TEST_TOKENIZER_ON_ORIGINAL_DATA(train_data, tokenized_train_data)
TEST_TOKENIZER_ON_ORIGINAL_DATA(test_data, tokenized_test_data)

# Find the max length of the tokenizers
MAX_LENGTH = -1
for inputs, outputs in tokenized_train_data:
    input_size = len(inputs)
    output_size = len(outputs)
    if input_size > MAX_LENGTH:
        MAX_LENGTH = input_size
    if output_size > MAX_LENGTH:
        MAX_LENGTH = output_size
print(MAX_LENGTH)


def generate_torch_vectors(tokenized_data, MAX_TOKEN_LIMIT):
    """
        This converts the vectors to torch vectors.
        We will have to implement padding so they are the same size
        For now I only keep those with a certain size.
    """
    filtered_data = []
    for input_vec, output_vec in tokenized_data:
        input_torch = torch.tensor(input_vec, dtype=torch.long, device=device).view(-1, 1)
        output_torch = torch.tensor(output_vec, dtype=torch.long, device=device).view(-1, 1)
        filtered_data.append([input_torch, output_torch])
    return filtered_data


torch_train = generate_torch_vectors(tokenized_train_data, MAX_LENGTH)
torch_validate = generate_torch_vectors(tokenized_test_data, MAX_LENGTH)

#unique_values =  get
