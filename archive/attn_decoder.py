import torch
from torch import nn
import torch.nn.functional as F
from utils import MAX_LENGTH

# This needs to be updated with https://bastings.github.io/annotated_encoder_decoder/
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderBahdanau(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.5, num_layers=2, model_type='lstm', attention=True):
        super(AttnDecoderBahdanau, self).__init__()

        # def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        #     super(AttnDecoderRNN, self).__init__()
        #     self.hidden_size = hidden_size
        #     self.output_size = output_size
        #     self.dropout_p = dropout_p
        #     self.max_length = max_length
        #
        #     self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        #     self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        #     self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        #     self.dropout = nn.Dropout(self.dropout_p)
        #     self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        #     self.out = nn.Linear(self.hidden_size, self.output_size)

        # self.model_type = model_type

        # Ref: https://www.youtube.com/watch?v=Qu81irGlR-0
        self.model_type = 'gru'
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.attention = attention

        self.embedding = nn.Embedding(output_size, hidden_size)

        self.wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wb = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        if self.model_type == 'lstm':
            self.recurrent = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout_p)
        else:
            self.recurrent = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        src_len = encoder_outputs.size(0)
        import pdb; pdb.set_trace()
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        output = embedded  # if we don't have attention
        # self.model_type == 'gru'

        # if self.attention:
        #     if self.model_type == 'lstm':
        #         energy = self.v(F.tanh(self.w1(hidden[0][-1]) + self.w2(encoder_outputs)))
        #     else:
        #         energy = self.v(F.tanh(self.w1(hidden[-1]) + self.w2(encoder_outputs)))

        # W_a(g_i − 1) + U_a(h_t)
        # previous decoder hidden state: g_i−1
        # encoder hidden state h_t
        import pdb; pdb.set_trace()
        vec_sum = self.wa(hidden[-1]) + self.wb(encoder_outputs)

        # e = v.tanh(W_a(g_i − 1) + U_a(h_t))
        align_score = self.v(F.tanh(vec_sum))

        # attn = softmax function (e)
        attn_w = F.softmax(align_score.view(1, -1), dim=1)

        #context vector = weighted sum of the encoder hidden states; encoder output in the query q
        context_vec = torch.matmul(attn_w, encoder_outputs)

        output = torch.cat((embedded[0], context_vec), 1)

        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        output, hidden = self.recurrent(output, hidden)
        output = self.out(output[0])

        if self.attention:
            return output, hidden, attn_w
        else:
            return output, hidden
