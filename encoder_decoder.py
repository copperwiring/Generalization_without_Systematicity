import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=self.n_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)


class AttentionUnit(nn.Module):
    def __init__(self, hidden_size, max_size):
        """
            Attention as introduced in the paper.
            Has 3 learnable parameters:

            U: Matrix that scales the previous decoder state (Hidden_Size x Hidden_Size)
            W: Matrix that scales the encoder states (Hidden_Size x Hidden_Size)
            v: Vector that scales the combined Hidden and Previous Encoder states

        """
        super(AttentionUnit, self).__init__()
        self.hidden_size = hidden_size

        # Initialize the parameters in torch so they are adjusted in the loss function
        # As Srishti pointed out, nn.Linear is just a simple transformation.
        # This makes it easier than manually setting parameters.

        self.U = torch.nn.Linear(self.hidden_size,
                                 self.hidden_size)  # nn.Parameter(torch.rand((hidden_size, hidden_size), dtype=torch.float32), requires_grad=True)
        self.W = torch.nn.Linear(self.hidden_size,
                                 self.hidden_size)  # nn.Parameter(torch.rand((hidden_size, hidden_size), dtype=torch.float32), requires_grad=True)
        self.v = torch.nn.Linear(self.hidden_size,
                                 1)  # nn.Parameter(torch.rand(hidden_size, dtype=torch.float32), requires_grad=True)

        # Tanh function that is applied
        self.tanh = nn.Tanh()

        # Softmax to calculate the attention.
        self.softmax = nn.Softmax(dim=0)

    def calculate_e_value(self, encoder_state, previous_decoder_state):
        # Previous decoder hidden states [-1], in case there is multiple layers
        # (Take the last layer). This might need to be checked.
        # Hidden x 1
        previous_hidden_scale = self.W(previous_decoder_state[-1])
        # Takes the encoder and returns Hidden x Max_Size
        encoder_state_scale = self.U(encoder_state)
        # Sum the vectors, this will broadcast. Hidden x Max_Size
        vecs_combines = previous_hidden_scale + encoder_state_scale
        # Tan Operation
        tan_op = self.tanh(vecs_combines)
        # Scale the vector to get a score e_score
        # E_score will comput the e value for each hidden state
        # returns Max_Size x 1
        e_score = self.v(tan_op)
        return e_score

    def forward(self, hidden, encoder_hidden):
        # Hidden is the previous decoder state
        # Econder hidden is all the hidden states of the decoder
        all_hidden_states_e = self.calculate_e_value(encoder_hidden, hidden)
        att = self.softmax(all_hidden_states_e)
        # Softmax the attention so it's a probability
        # att is of the size Max_Size x 1
        context_vec = att * encoder_hidden
        # return Max_Size x Hidden
        # Scale the encoder states by the softmaxed attention and then
        # comput the final context vector by summing it.
        context_vec = context_vec.sum(axis=0)
        # Flatten the att array to size Max_Size
        # Context Vector is of size Max_Size, Hidden
        return att.flatten(), context_vec

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttentionDecoderGRU(nn.Module):
    """
        This modules uses concatenation to take the context vector and encoding
        vector to GRU unit. This means the GRU will work on a vector which has all the features
        and the mechanims inside the GRU unit (Reset Gates, Update Gates) will be tuned on both
        the information (without any alterations.). This means the GRU unit needs to take inputs
        of twice the size(2*h), and still outputs a hidden size of h.

        The key is that the ContextVector is used in both the inputs of the GRU and also in the
        prediction of the outputs.
    """

    def __init__(self, hidden_size, output_size, dropout_rate_emb, dropout_seq, max_length, n_layers=1):
        super(AttentionDecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.MAX_SIZE = max_length
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention_layer = AttentionUnit(self.hidden_size, self.MAX_SIZE)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, dropout=dropout_seq, num_layers=self.n_layers)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout_rate_emb)

    def forward(self, input, prev_hidden, encoder_hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = self.dropout(output)
        output = F.relu(output)

        # Decoder needs to calculate the attention (Read the comments in the AttentionUnit class)
        att_vec, context_vec = self.attention_layer(prev_hidden, encoder_hidden)

        # Combine the Context Vextor with the Output (of the embeddings)
        output = torch.cat((output[0], context_vec.view(1, -1)), 1)

        output, new_hidden = self.gru(output.view(1, 1, -1), prev_hidden)
        # Calculate the hidden state of the Unit.
        # The paper then states that this hidden state is concatenated to the
        # context vector and this is used to predict the next output action (a_i)
        # The tutorial used a log_softmax (likely to avoid overflow)
        output = torch.cat((output[-1], context_vec.view(1, -1)), dim=1)
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        return output, new_hidden, att_vec

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)


class AttentionDecoderGRUv2(nn.Module):
    """
        This modules uses a linear combination to take the context vector and encoding
        vector to the hidden dimension. This means the GRU will work on a vector which
        has already the combination of the information, and the mechanisms in the seqModel
        will be in this linear combination inputs. This is more similar to what was originally
        shown in the tutorial.
    """

    def __init__(self, hidden_size, output_size, dropout_rate_emb, dropout_seq, max_length, n_layers=1):
        super(AttentionDecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.MAX_SIZE = max_length
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention_layer = AttentionUnit(self.hidden_size, self.MAX_SIZE)
        self.combine_context_emb = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, dropout=dropout_seq, num_layers=self.n_layers)
        self.out = nn.Linear(self.hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout_rate_emb)

    def forward(self, input, prev_hidden, encoder_hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = self.dropout(output)
        output = F.relu(output)

        # Decoder needs to calculate the attention (Read the comments in the AttentionUnit class)
        att_vec, context_vec = self.attention_layer(prev_hidden, encoder_hidden)

        # Combine the Context Vextor with the Output (of the embeddings)
        output = torch.cat((output, context_vec.unsqueeze(0)), 1)
        output = self.combine_context_emb(output)

        output, new_hidden = self.gru(output, prev_hidden)
        # Calculate the hidden state of the Unit.
        # The paper then states that this hidden state is concatenated to the
        # context vector and this is used to predict the next output action (a_i)
        # The tutorial used a log_softmax (likely to avoid overflow)
        output = torch.cat((output[-1], context_vec.view(1, -1)), 1)
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        return output, new_hidden, att_vec

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)


class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_rate_emb, dropout_seq, max_length, n_layers=1):
        """
            The DecoderGRU has the same call signature as the AttentionDecoderGRU so we can slot in depending
            on what we want to test.
        """
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.MAX_SIZE = max_length
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        # self.attention_layer = AttentionUnit(self.hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, dropout=dropout_seq, num_layers=self.n_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate_emb)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, prev_hidden, encoder_hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output = self.dropout(output)
        # att_vec, context_vec = self.attention_layer(prev_hidden, encoder_hidden, encoder_length)
        att_vec = torch.zeros(self.MAX_SIZE)
        output, new_hidden = self.rnn(output, prev_hidden)
        output = self.softmax(self.out(output[0]))
        return output, new_hidden, att_vec

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)