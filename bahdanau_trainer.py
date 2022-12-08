import os.path

import matplotlib.pyplot as plt
import torch, time, random
from torch import optim
from utils import *
import numpy as np
import matplotlib as mpl
import torch.nn as nn
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BahDanauTrainer():
    """
        This class contains the same methods as in the tutorial, but it adjusts them
        so the parameters passed are the ones expected by the implementation.

        This also allows to have more than one implementation of the train/trainiters/evalmethod.
    """

    def train(input_tensor, target_tensor, tokenizer,
              encoder, decoder, encoder_optimizer,
              decoder_optimizer, criterion, max_length):
        teacher_forcing_ratio = 0.5
        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[tokenizer.SOS_TOKEN]], device=device)

        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length): # input, prev_hidden, encoder_hidden
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += criterion(decoder_output, target_tensor[di])
                # if decoder_input.item() == tokenizer.EOS_TOKEN or decoder_input.item() == tokenizer.PAD_TOKEN:
                #     break
                if decoder_input.item() == tokenizer.EOS_TOKEN:
                    break

        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5)

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def train_loop(encoder, decoder, tokenizer, data, n_iters, exp_result_dir, seed_value=1, print_every=1000, plot_every=100, learning_rate=0.001, max_length=50):
        start = time.time()
        mode = 'train'
        # creating a new directory where the loss plots will be saved
        plot_dir = "plot_loss"
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)  # Updated
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)  # Updated
        training_pairs = [random.choice(data)
                          for i in range(n_iters)]
        criterion = nn.NLLLoss()

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            loss = BahDanauTrainer.train(input_tensor, target_tensor, tokenizer, encoder,
                                         decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        # save the trained model weights for a final time
        # save_model(epochs, model, optim, criterion)

        loss_dir = "plot_loss"
        Path(loss_dir).mkdir(parents=True, exist_ok=True)
        showPlot(plot_losses, os.path.join(loss_dir, exp_result_dir + "_seed" + str(seed_value) + "_" + mode))

    def evaluate(encoder, decoder, tokenizer, token_vector, max_length=50):
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
                topv, topi = decoder_output.data.topk(1)
                decoder_attentions[di] = decoder_attention
                if topi.item() == tokenizer.EOS_TOKEN:
                    decoded_words.append('EOS')
                    break
                else:
                    decoded_words.append(tokenizer.n_to_command[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]

    def evaluateToken(encoder, decoder, tokenizer, token_vector, max_length=50):
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
                topv, topi = decoder_output.data.topk(1)
                decoder_attentions[di] = decoder_attention
                if topi.item() == tokenizer.EOS_TOKEN:
                    decoded_words.append('EOS')
                    break
                else:
                    decoded_words.append(topi.item())

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]

    def getDatasetAccuracy(encoder, decoder, tokenizer, test_data):
        # import pdb; pdb.set_trace()
        correct_array = np.zeros(len(test_data), dtype=bool)
        for i in range(len(test_data)):
            output_words, attentions = BahDanauTrainer.evaluate(encoder, decoder, tokenizer, test_data[i][0])
            y = tokenizer.decode_outputs(test_data[i][1].flatten().detach().to("cpu").numpy())
            y_pred = output_words
            correct_array[i] = y == y_pred
        return np.array(correct_array).sum() / len(correct_array)

    def evaluateRandomly(encoder, decoder, tokenizer, data, exp_result_dir, mode, n=10, plot_attention=False):
        for i in range(n):
            i_pair = random.choice(np.arange(len(data)))
            vector_detached = data[i_pair]
            inputs = tokenizer.decode_inputs(vector_detached[0].flatten().detach().to("cpu").numpy())
            outputs_expect = tokenizer.decode_outputs(vector_detached[1].flatten().detach().to("cpu").numpy())
            # print('>', " ".join(inputs))
            # print('=', " ".join(outputs_expect))
            output_words, attentions = BahDanauTrainer.evaluate(encoder, decoder, tokenizer, data[i_pair][0])
            output_sentence = ' '.join(output_words)
            output_expected_sent = " ".join(outputs_expect)
            # print("Attention: ", torch.round(attentions[:len(output_words),:len(data[i_pair][0])],decimals=5))
            # print('<', output_sentence)
            if plot_attention:
                att_relevant_data = attentions[:len(output_words), :len(data[i_pair][0])]
                fig, ax = plt.subplots()
                if output_sentence == output_expected_sent:
                    im = ax.imshow(att_relevant_data.T, cmap=mpl.colormaps["viridis"])
                else:
                    im = ax.imshow(att_relevant_data.T, cmap=mpl.colormaps["magma"])
                # Show all ticks and label them with the respective list entries
                ax.set_yticks(np.arange(len(data[i_pair][0])), labels=inputs, rotation=45, ha='right')
                ax.set_xticks(np.arange(len(output_words)), labels=output_words)
                plt.xticks(rotation=90)
                plt.show()
                # creating a directory where the results will be saved
                attn_dir = "attn_visuals"
                Path(attn_dir).mkdir(parents=True, exist_ok=True)
                # Path(exp_result_dir).mkdir(parents=True, exist_ok=True)

                plt.savefig((os.path.join(attn_dir, exp_result_dir + "_seed"+ str(i) + "_attn_" + mode + ".png")))
            # print('Correct: ', output_sentence == output_expected_sent)
