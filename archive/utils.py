import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import math
from dataloader import TokenizerSCAN
from load_data import load_data



# Load train and test data from the base directory with the type of file
# TO DO: add tye of file in the func
train_data = load_data("train")
test_data = load_data("test")
hidden_size = 200  # As per the paper
dropout_prob = 0.5  # As per the paper
MAX_LENGTH = -1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tokenizer():
    return TokenizerSCAN()

tokenizer = get_tokenizer()

def tokenize_data(data):
    token_data = []
    for i in range(len(data)):
        input_line, output_line = data[i]
        tokenized_input, tokenized_output = tokenizer.encode_inputs(input_line), tokenizer.encode_outputs(output_line)
        token_data.append([tokenized_input, tokenized_output])
    return token_data
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def tensorsFromPair(pair):
    return (pair[0], pair[1])


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# Plot training




def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

# def tokenized_data(sample = "true"):
#     return tokenized_data(sample)

# def tokenized_data(train=True):
#     if train:
#         return tokenize_data(train_data)
#     else:
#         return tokenize_data(test_data)
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

def get_torched_vec_data(tokenized_data):
    return generate_torch_vectors(tokenized_data, MAX_LENGTH)



