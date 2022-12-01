import numpy as np
import random
import torch


def evaluate(encoder, decoder, token_vector, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = token_vector
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[tokenizer.SOS_TOKEN]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == tokenizer.EOS_TOKEN:
                decoded_words.append('EOS')
                break
            else:
                decoded_words.append(tokenizer.n_to_command[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, data, n=5):
    for i in range(n):
        i_pair = random.choice(np.arange(len(data)))
        vector_detached = data[i_pair]
        print('>', " ".join(tokenizer.decode_inputs(vector_detached[0].flatten().detach().to("cpu").numpy())))
        print('=', " ".join(tokenizer.decode_outputs(vector_detached[1].flatten().detach().to("cpu").numpy())))
        output_words, attentions = evaluate(encoder, decoder, data[i_pair][0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def getDatasetAccuracy(encoder, decoder, test_data):
    correct_array = np.zeros(len(test_data), dtype=bool)
    for i in range(len(test_data)):
        output_words, attentions = evaluate(encoder, decoder, test_data[i][0])
        y = tokenizer.decode_outputs(test_data[i][1].flatten().detach().to("cpu").numpy())
        y_pred = output_words
        correct_array[i] = y == y_pred
    return np.array(correct_array).sum() / len(correct_array)
