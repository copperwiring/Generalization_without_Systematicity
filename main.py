
from __future__ import unicode_literals, print_function, division

import os.path

import torch, random
from torch.utils.data import Dataset
from loader import SCANDataset, TokenizerSCAN, SCANDataLoader
from encoder_decoder import EncoderGRU, AttentionDecoderGRU
from bahdanau_trainer import BahDanauTrainer
from torch.utils.data import Subset
import pandas as pd
from pathlib import Path

# Loading the data and training the encoder
DATA_CLOUD = "SCAN/SCAN-master/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class SCANDataset(Dataset):
#     def __read_each_line(filename):
#         """Read the file line by line and split into input and output"""
#         # Read the file and split into lines
#         with open(filename, encoding='utf-8') as file:
#             lines = [line.rstrip('\n') for line in file]
#             keyword = ' OUT: '
#             partition_lines = [line.partition(keyword) for line in
#                                lines]  # Split sentences before and after "OUT" keyword
#             input_part = [line[0].replace("IN: ", "") for line in partition_lines]
#             output_part = [line[2] for line in partition_lines]
#
#         return input_part, output_part
#
#     """the function is initialised here"""
#
#     def __init__(self, root='SCAN', split_type='simple', target_file='simple', train=True):
#         self.train = train
#         self.task = target_file
#         if self.train:
#             self.data_file = os.path.join(root, split_type, f'tasks_train_{self.task}.txt')
#         else:
#             self.data_file = os.path.join(root, split_type, f'tasks_test_{self.task}.txt')
#         print("Loading datafile: ", self.data_file)
#         self.input_data, self.output_data = SCANDataset.__read_each_line(self.data_file)
#
#     def __getitem__(self, index):
#         """gives one item at a time"""
#         input_data_idx = self.input_data[index]
#         output_data_idx = self.output_data[index]
#
#         return input_data_idx, output_data_idx
#
#     def __len__(self):
#         """the function returns length of data"""
#         return len(self.input_data)
#
#
# class TokenizerSCAN():
#     """
#         dataset:SCANDatasetLoader Loaded data with the Torch Dataset
#     """
#
#     def __init__(self):
#         self.SOS_TOKEN = 0
#         self.EOS_TOKEN = 1
#
#         self.word_to_n = {"EOS": self.EOS_TOKEN, "SOS": self.SOS_TOKEN}
#         self.n_to_word = dict()
#         self.command_to_n = {"EOS": self.EOS_TOKEN, "SOS": self.SOS_TOKEN}
#         self.n_to_command = dict()
#         self.total_word = len(self.word_to_n)
#         self.total_command = len(self.command_to_n)
#
#     def get_n_words(self):
#         return len(set(self.n_to_word.keys()))
#
#     def get_n_cmds(self):
#         return len(set(self.n_to_command.keys()))
#
#     def fit(self, dataset: SCANDataset):
#         for i in range(len(dataset)):
#             input_line, output_line = dataset[i]
#             for word in input_line.split(" "):
#                 if word not in self.word_to_n:
#                     self.word_to_n[word] = self.total_word
#                     self.total_word += 1
#             for cmd in output_line.split(" "):
#                 if cmd not in self.command_to_n:
#                     self.command_to_n[cmd] = self.total_command
#                     self.total_command += 1
#         # Create the Reverse dictionary
#         self.n_to_word = {v: k for k, v in self.word_to_n.items()}
#         self.n_to_command = {v: k for k, v in self.command_to_n.items()}
#         print(f"Total of {self.get_n_words()} words and {self.get_n_cmds()} commands tokenized.")
#
#     def encode_inputs(self, words):
#         if type(words) != list:
#             if type(words) == str:
#                 words = words.split(" ")
#         return ([self.word_to_n[word] for word in words] + [self.word_to_n["EOS"]])
#
#     def encode_outputs(self, cmds):
#         if type(cmds) != list:
#             if type(cmds) == str:
#                 cmds = cmds.split(" ")
#         return ([self.command_to_n[cmd] for cmd in cmds] + [self.command_to_n["EOS"]])
#
#     def decode_inputs(self, words_n):
#         if type(words_n) != list:
#             words_n = list(words_n)
#         return [self.n_to_word[word_n] for word_n in words_n]
#
#     def decode_outputs(self, cmds_n):
#         if type(cmds_n) != list:
#             cmds_n = list(cmds_n)
#         return [self.n_to_command[cmd_n] for cmd_n in cmds_n]


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

# Dataloader
# Take a "Dataset" + "Tokenizer" (fit to the dataset) and generates Tensors to be fed into the models.
# class SCANDataLoader(Dataset):
#     def __init__(self, dataset:SCANDataset, tokenizer):
#         self.dataset = dataset
#         self.tokenizer = tokenizer
#
#     def __getitem__(self, index):
#         """gives one item at a time"""
#         input_sent, output_sent = self.dataset[index]
#         tokenized_inputs = self.tokenizer.encode_inputs(input_sent)
#         tokenized_outputs = self.tokenizer.encode_outputs(output_sent)
#         input_data = torch.tensor(tokenized_inputs, dtype=torch.long, device=device).view(-1, 1)
#         output_data = torch.tensor(tokenized_outputs, dtype=torch.long, device=device).view(-1, 1)
#         return input_data, output_data
#
#     def __len__(self):
#         """the function returns length of data"""
#         return len(self.dataset)

tensor_train_data = SCANDataLoader(train_data, tokenizer)
tensor_test_data = SCANDataLoader(test_data, tokenizer)


# takes 100 samples
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


# MAX_LENGTH = 50

#
# class EncoderGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, n_layers=1):
#         super(EncoderGRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size, num_layers=self.n_layers)
#
#     def forward(self, input, hidden):
#         embedded = self.embedding(input).view(1, 1, -1)
#         output = embedded
#         output, hidden = self.gru(output, hidden)
#         return output, hidden
#
#     def initHidden(self):
#         return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)
#
#
# class AttentionUnit(nn.Module):
#     def __init__(self, hidden_size, max_size):
#         """
#             Attention as introduced in the paper.
#             Has 3 learnable parameters:
#
#             U: Matrix that scales the previous decoder state (Hidden_Size x Hidden_Size)
#             W: Matrix that scales the encoder states (Hidden_Size x Hidden_Size)
#             v: Vector that scales the combined Hidden and Previous Encoder states
#
#         """
#         super(AttentionUnit, self).__init__()
#         self.hidden_size = hidden_size
#
#         # Initialize the parameters in torch so they are adjusted in the loss function
#         # As Srishti pointed out, nn.Linear is just a simple transformation.
#         # This makes it easier than manually setting parameters.
#
#         self.U = torch.nn.Linear(self.hidden_size,
#                                  self.hidden_size)  # nn.Parameter(torch.rand((hidden_size, hidden_size), dtype=torch.float32), requires_grad=True)
#         self.W = torch.nn.Linear(self.hidden_size,
#                                  self.hidden_size)  # nn.Parameter(torch.rand((hidden_size, hidden_size), dtype=torch.float32), requires_grad=True)
#         self.v = torch.nn.Linear(self.hidden_size,
#                                  1)  # nn.Parameter(torch.rand(hidden_size, dtype=torch.float32), requires_grad=True)
#
#         # Tanh function that is applied
#         self.tanh = nn.Tanh()
#
#         # Softmax to calculate the attention.
#         self.softmax = nn.Softmax(dim=0)
#
#     def calculate_e_value(self, encoder_state, previous_decoder_state):
#         # Previous decoder hidden states [-1], in case there is multiple layers
#         # (Take the last layer). This might need to be checked.
#         # Hidden x 1
#         previous_hidden_scale = self.W(previous_decoder_state[-1])
#         # Takes the encoder and returns Hidden x Max_Size
#         encoder_state_scale = self.U(encoder_state)
#         # Sum the vectors, this will broadcast. Hidden x Max_Size
#         vecs_combines = previous_hidden_scale + encoder_state_scale
#         # Tan Operation
#         tan_op = self.tanh(vecs_combines)
#         # Scale the vector to get a score e_score
#         # E_score will comput the e value for each hidden state
#         # returns Max_Size x 1
#         e_score = self.v(tan_op)
#         return e_score
#
#     def forward(self, hidden, encoder_hidden):
#         # Hidden is the previous decoder state
#         # Econder hidden is all the hidden states of the decoder
#         all_hidden_states_e = self.calculate_e_value(encoder_hidden, hidden)
#         att = self.softmax(all_hidden_states_e)
#         # Softmax the attention so it's a probability
#         # att is of the size Max_Size x 1
#         context_vec = att * encoder_hidden
#         # return Max_Size x Hidden
#         # Scale the encoder states by the softmaxed attention and then
#         # comput the final context vector by summing it.
#         context_vec = context_vec.sum(axis=0)
#         # Flatten the att array to size Max_Size
#         # Context Vector is of size Max_Size, Hidden
#         return att.flatten(), context_vec
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
#
#
# class AttentionDecoderGRU(nn.Module):
#     """
#         This modules uses concatenation to take the context vector and encoding
#         vector to GRU unit. This means the GRU will work on a vector which has all the features
#         and the mechanims inside the GRU unit (Reset Gates, Update Gates) will be tuned on both
#         the information (without any alterations.). This means the GRU unit needs to take inputs
#         of twice the size(2*h), and still outputs a hidden size of h.
#
#         The key is that the ContextVector is used in both the inputs of the GRU and also in the
#         prediction of the outputs.
#     """
#
#     def __init__(self, hidden_size, output_size, dropout_rate_emb, dropout_seq, max_length, n_layers=1):
#         super(AttentionDecoderGRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.MAX_SIZE = max_length
#         self.n_layers = n_layers
#
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.attention_layer = AttentionUnit(self.hidden_size, self.MAX_SIZE)
#         self.gru = nn.GRU(hidden_size * 2, hidden_size, dropout=dropout_seq, num_layers=self.n_layers)
#         self.out = nn.Linear(hidden_size * 2, output_size)
#         self.dropout = nn.Dropout(dropout_rate_emb)
#
#     def forward(self, input, prev_hidden, encoder_hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = self.dropout(output)
#         output = F.relu(output)
#
#         # Decoder needs to calculate the attention (Read the comments in the AttentionUnit class)
#         att_vec, context_vec = self.attention_layer(prev_hidden, encoder_hidden)
#
#         # Combine the Context Vextor with the Output (of the embeddings)
#         output = torch.cat((output[0], context_vec.view(1, -1)), 1)
#
#         output, new_hidden = self.gru(output.view(1, 1, -1), prev_hidden)
#         # Calculate the hidden state of the Unit.
#         # The paper then states that this hidden state is concatenated to the
#         # context vector and this is used to predict the next output action (a_i)
#         # The tutorial used a log_softmax (likely to avoid overflow)
#         output = torch.cat((new_hidden[-1], context_vec.view(1, -1)), dim=1)
#         output = self.out(output)
#         output = F.log_softmax(output, dim=1)
#         return output, new_hidden, att_vec
#
#     def initHidden(self):
#         return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)
#
#
# class AttentionDecoderGRUv2(nn.Module):
#     """
#         This modules uses a linear combination to take the context vector and encoding
#         vector to the hidden dimension. This means the GRU will work on a vector which
#         has already the combination of the information, and the mechanisms in the seqModel
#         will be in this linear combination inputs. This is more similar to what was originally
#         shown in the tutorial.
#     """
#
#     def __init__(self, hidden_size, output_size, dropout_rate_emb, dropout_seq, max_length, n_layers=1):
#         super(AttentionDecoderGRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.MAX_SIZE = max_length
#         self.n_layers = n_layers
#
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.attention_layer = AttentionUnit(self.hidden_size, self.MAX_SIZE)
#         self.combine_context_emb = nn.Linear(2 * self.hidden_size, self.hidden_size)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size, dropout=dropout_seq, num_layers=self.n_layers)
#         self.out = nn.Linear(self.hidden_size * 2, output_size)
#         self.dropout = nn.Dropout(dropout_rate_emb)
#
#     def forward(self, input, prev_hidden, encoder_hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = self.dropout(output)
#         output = F.relu(output)
#
#         # Decoder needs to calculate the attention (Read the comments in the AttentionUnit class)
#         att_vec, context_vec = self.attention_layer(prev_hidden, encoder_hidden)
#
#         # Combine the Context Vextor with the Output (of the embeddings)
#         output = torch.cat((output, context_vec.unsqueeze(0)), 1)
#         output = self.combine_context_emb(output)
#
#         output, new_hidden = self.gru(output, prev_hidden)
#         # Calculate the hidden state of the Unit.
#         # The paper then states that this hidden state is concatenated to the
#         # context vector and this is used to predict the next output action (a_i)
#         # The tutorial used a log_softmax (likely to avoid overflow)
#         output = torch.cat((new_hidden[-1], context_vec.view(1, -1)), 1)
#         output = self.out(output)
#         output = F.log_softmax(output, dim=1)
#         return output, new_hidden, att_vec
#
#     def initHidden(self):
#         return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)
#
#
# class DecoderGRU(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_rate_emb, dropout_seq, max_length, n_layers=1):
#         """
#             The DecoderGRU has the same call signature as the AttentionDecoderGRU so we can slot in depending
#             on what we want to test.
#         """
#         super(DecoderGRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.MAX_SIZE = max_length
#         self.n_layers = n_layers
#
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         # self.attention_layer = AttentionUnit(self.hidden_size)
#         self.rnn = nn.GRU(hidden_size, hidden_size, dropout=dropout_seq, num_layers=self.n_layers)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.dropout = nn.Dropout(dropout_rate_emb)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, input, prev_hidden, encoder_hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         output = self.dropout(output)
#         # att_vec, context_vec = self.attention_layer(prev_hidden, encoder_hidden, encoder_length)
#         att_vec = torch.zeros(self.MAX_SIZE)
#         output, new_hidden = self.rnn(output, prev_hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, new_hidden, att_vec
#
#     def initHidden(self):
#         return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)


# class BahDanauTrainer():
#     """
#         This class contains the same methods as in the tutorial, but it adjusts them
#         so the parameters passed are the ones expected by the implementation.
#
#         This also allows to have more than one implementation of the train/trainiters/evalmethod.
#     """
#
#     def train(input_tensor, target_tensor, tokenizer,
#               encoder, decoder, encoder_optimizer,
#               decoder_optimizer, criterion, max_length=MAX_LENGTH):
#         teacher_forcing_ratio = 0.5
#         encoder_hidden = encoder.initHidden()
#
#         encoder_optimizer.zero_grad()
#         decoder_optimizer.zero_grad()
#
#         input_length = input_tensor.size(0)
#         target_length = target_tensor.size(0)
#         encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
#
#         loss = 0
#
#         for ei in range(input_length):
#             encoder_output, encoder_hidden = encoder(
#                 input_tensor[ei], encoder_hidden)
#             encoder_outputs[ei] = encoder_output[0, 0]
#
#         decoder_input = torch.tensor([[tokenizer.SOS_TOKEN]], device=device)
#
#         decoder_hidden = encoder_hidden
#         use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
#
#         if use_teacher_forcing:
#             # Teacher forcing: Feed the target as the next input
#             for di in range(target_length):
#                 decoder_output, decoder_hidden, decoder_attention = decoder(
#                     decoder_input, decoder_hidden, encoder_outputs)
#                 loss += criterion(decoder_output, target_tensor[di])
#                 decoder_input = target_tensor[di]  # Teacher forcing
#
#         else:
#             # Without teacher forcing: use its own predictions as the next input
#             for di in range(target_length):
#                 decoder_output, decoder_hidden, decoder_attention = decoder(
#                     decoder_input, decoder_hidden, encoder_outputs)
#                 topv, topi = decoder_output.topk(1)
#                 decoder_input = topi.squeeze().detach()  # detach from history as input
#                 loss += criterion(decoder_output, target_tensor[di])
#                 if decoder_input.item() == tokenizer.EOS_TOKEN or decoder_input.item() == tokenizer.PAD_TOKEN:
#                     break
#
#         loss.backward()
#
#         torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
#         torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5)
#
#         encoder_optimizer.step()
#         decoder_optimizer.step()
#
#         return loss.item() / target_length
#
#     def train_loop(encoder, decoder, tokenizer, data, n_iters, print_every=1000, plot_every=100, learning_rate=0.001):
#         start = time.time()
#         plot_losses = []
#         print_loss_total = 0  # Reset every print_every
#         plot_loss_total = 0  # Reset every plot_every
#
#         encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)  # Updated
#         decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)  # Updated
#         training_pairs = [random.choice(data)
#                           for i in range(n_iters)]
#         criterion = nn.NLLLoss()
#
#         for iter in range(1, n_iters + 1):
#             training_pair = training_pairs[iter - 1]
#             input_tensor = training_pair[0]
#             target_tensor = training_pair[1]
#             loss = BahDanauTrainer.train(input_tensor, target_tensor, tokenizer, encoder,
#                                          decoder, encoder_optimizer, decoder_optimizer, criterion)
#             print_loss_total += loss
#             plot_loss_total += loss
#
#             if iter % print_every == 0:
#                 print_loss_avg = print_loss_total / print_every
#                 print_loss_total = 0
#                 print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
#                                              iter, iter / n_iters * 100, print_loss_avg))
#
#             if iter % plot_every == 0:
#                 plot_loss_avg = plot_loss_total / plot_every
#                 plot_losses.append(plot_loss_avg)
#                 plot_loss_total = 0
#
#         showPlot(plot_losses)
#
#     def evaluate(encoder, decoder, tokenizer, token_vector, max_length=MAX_LENGTH):
#         with torch.no_grad():
#             input_tensor = token_vector
#             input_length = input_tensor.size()[0]
#             encoder_hidden = encoder.initHidden()
#
#             encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
#
#             for ei in range(input_length):
#                 encoder_output, encoder_hidden = encoder(input_tensor[ei],
#                                                          encoder_hidden)
#                 encoder_outputs[ei] += encoder_output[0, 0]
#
#             decoder_input = torch.tensor([[tokenizer.SOS_TOKEN]], device=device)  # SOS
#             decoder_hidden = encoder_hidden
#
#             decoded_words = []
#             decoder_attentions = torch.zeros(max_length, max_length)
#             for di in range(max_length):
#                 decoder_output, decoder_hidden, decoder_attention = decoder(
#                     decoder_input, decoder_hidden, encoder_outputs)
#                 topv, topi = decoder_output.data.topk(1)
#                 decoder_attentions[di] = decoder_attention
#                 if topi.item() == tokenizer.EOS_TOKEN:
#                     decoded_words.append('EOS')
#                     break
#                 else:
#                     decoded_words.append(tokenizer.n_to_command[topi.item()])
#
#                 decoder_input = topi.squeeze().detach()
#
#             return decoded_words, decoder_attentions[:di + 1]
#
#     def evaluateToken(encoder, decoder, tokenizer, token_vector, max_length=MAX_LENGTH):
#         with torch.no_grad():
#             input_tensor = token_vector
#             input_length = input_tensor.size()[0]
#             encoder_hidden = encoder.initHidden()
#
#             encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
#
#             for ei in range(input_length):
#                 encoder_output, encoder_hidden = encoder(input_tensor[ei],
#                                                          encoder_hidden)
#                 encoder_outputs[ei] += encoder_output[0, 0]
#
#             decoder_input = torch.tensor([[tokenizer.SOS_TOKEN]], device=device)  # SOS
#             decoder_hidden = encoder_hidden
#
#             decoded_words = []
#             decoder_attentions = torch.zeros(max_length, max_length)
#             for di in range(max_length):
#                 decoder_output, decoder_hidden, decoder_attention = decoder(
#                     decoder_input, decoder_hidden, encoder_outputs)
#                 topv, topi = decoder_output.data.topk(1)
#                 decoder_attentions[di] = decoder_attention
#                 if topi.item() == tokenizer.EOS_TOKEN:
#                     decoded_words.append('EOS')
#                     break
#                 else:
#                     decoded_words.append(topi.item())
#
#                 decoder_input = topi.squeeze().detach()
#
#             return decoded_words, decoder_attentions[:di + 1]
#
#     def getDatasetAccuracyOwn(encoder, decoder, tokenizer, test_data):
#         correct_array = np.zeros(len(test_data), dtype=bool)
#         for i in range(len(test_data)):
#             output_words, attentions = BahDanauTrainer.evaluate(encoder, decoder, tokenizer, test_data[i][0])
#             y = tokenizer.decode_outputs(test_data[i][1].flatten().detach().to("cpu").numpy())
#             y_pred = output_words
#             correct_array[i] = y == y_pred
#         return np.array(correct_array).sum() / len(correct_array)
#
#     def evaluateRandomly(encoder, decoder, tokenizer, data, n=10, plot_attention=False):
#         for i in range(n):
#             i_pair = random.choice(np.arange(len(data)))
#             vector_detached = data[i_pair]
#             inputs = tokenizer.decode_inputs(vector_detached[0].flatten().detach().to("cpu").numpy())
#             outputs_expect = tokenizer.decode_outputs(vector_detached[1].flatten().detach().to("cpu").numpy())
#             print('>', " ".join(inputs))
#             print('=', " ".join(outputs_expect))
#             output_words, attentions = BahDanauTrainer.evaluate(encoder, decoder, tokenizer, data[i_pair][0])
#             output_sentence = ' '.join(output_words)
#             output_expected_sent = " ".join(outputs_expect)
#             # print("Attention: ", torch.round(attentions[:len(output_words),:len(data[i_pair][0])],decimals=5))
#             print('<', output_sentence)
#             if plot_attention:
#                 att_relevant_data = attentions[:len(output_words), :len(data[i_pair][0])]
#                 fig, ax = plt.subplots()
#                 if output_sentence == output_expected_sent:
#                     im = ax.imshow(att_relevant_data.T, cmap=mpl.colormaps["viridis"])
#                 else:
#                     im = ax.imshow(att_relevant_data.T, cmap=mpl.colormaps["magma"])
#                 # Show all ticks and label them with the respective list entries
#                 ax.set_yticks(np.arange(len(data[i_pair][0])), labels=inputs, rotation=45, ha='right')
#                 ax.set_xticks(np.arange(len(output_words)), labels=output_words)
#                 plt.xticks(rotation=90)
#                 plt.show()
#             print('Correct: ', output_sentence == output_expected_sent)

# Example of running the network.

# train_smaller =

hidden_size = 100
dropout_prob = 0.1
subset=True

for i in range(5):
    results_dict = {"train": [], "test": []}
    print(f"Seed: {i + 1}")
    seed_val = i+1
    random.seed(i)
    encoder2 = EncoderGRU(tokenizer.get_n_words(), hidden_size).to(device)
    decoder2 = AttentionDecoderGRU(hidden_size, tokenizer.get_n_cmds(), dropout_prob, dropout_prob, MAX_LENGTH).to(device)
    if subset:
        BahDanauTrainer.train_loop(encoder2, decoder2, tokenizer, tensor_subset_train_data, 50, seed_val, print_every=10, plot_every=5, max_length=MAX_LENGTH)
        results_dict['train'].append(BahDanauTrainer.getDatasetAccuracy(encoder2, decoder2, tokenizer, tensor_subset_train_data))
        results_dict['test'].append(BahDanauTrainer.getDatasetAccuracy(encoder2, decoder2, tokenizer, tensor_subset_test_data))
        #BahDanauTrainer.evaluateRandomly(encoder2, decoder2, tokenizer, tensor_subset_test_data, mode="test", n=10, plot_attention=True)
    else:
        BahDanauTrainer.train_loop(encoder2, decoder2, tokenizer, tensor_train_data, 5000, print_every=50, plot_every=500, max_length=MAX_LENGTH)
        results_dict['train'].append(BahDanauTrainer.getDatasetAccuracy(encoder2, decoder2, tensor_train_data))
        results_dict['test'].append(BahDanauTrainer.getDatasetAccuracy(encoder2, decoder2, tensor_test_data))

    results_pd = pd.DataFrame.from_dict(results_dict) # fix because they aint in diff columns
    # creating a directory where the results will be saved
    results_dir = "results"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_pd.to_csv(os.path.join(results_dir, str(seed_val) + "_" + "results.csv"))