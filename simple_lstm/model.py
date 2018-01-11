# coding=utf8
import json
import random
import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from custom_token import *

with open('config.json') as config_file:
    config = json.load(config_file)
USE_CUDA = config['TRAIN']['CUDA']
def create_seq2seq(vocab_size):
    hidden_size = config['MODEL']['HIDDEN_SIZE']
    attn_method = config['MODEL']['ATTN_METHOD']
    n_encoder_layers = config['MODEL']['N_ENCODER_LAYERS']
    n_decoder_layers = config['MODEL']['N_DECODER_LAYERS']
    dropout = config['MODEL']['DROPOUT']
    encoder = Encoder(vocab_size, hidden_size, n_layers = n_encoder_layers, dropout=dropout)
    decoder = Decoder(hidden_size, vocab_size, n_layers = n_decoder_layers, dropout=dropout)
    return Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        max_length=config['LOADER']['MAX_LENGTH']
    )

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, max_length=20):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length

        if USE_CUDA:
            self.cuda()

        for param in self.parameters():
            param.data.uniform_(-0.08, 0.08)

    def forward(self, input_group, target_group=(None, None),
            teacher_forcing_ratio=0.5, is_train = True):
        input_var, input_lens = input_group
        encoder_outputs, (h_t, c_t) = self.encoder(input_var, input_lens)

        batch_size = input_var.size(1)
        target_var, target_lens = target_group
        if target_var is None or target_lens is None:
            max_target_length = self.max_length
            teacher_forcing_ratio = 0 # without teacher forcing
        else:
            max_target_length = max(target_lens)

        # store all decoder outputs
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, self.decoder.output_size))
        # first decoder input
        decoder_input = Variable(torch.LongTensor([GO_token] * batch_size), requires_grad=False)
        decoder_h_t = Variable(torch.zeros(self.decoder.n_layers,batch_size, self.decoder.hidden_size))
        decoder_h_t[0] = h_t[-1]
        decoder_c_t = Variable(torch.zeros(self.decoder.n_layers,batch_size, self.decoder.hidden_size))

        if USE_CUDA:
            all_decoder_outputs.data = all_decoder_outputs.data.cuda()
            decoder_input.data = decoder_input.data.cuda()
            decoder_h_t.data = decoder_h_t.data.cuda()
            decoder_c_t.data = decoder_c_t.data.cuda()

        decoder_hidden = (decoder_h_t, decoder_c_t)
        for t in range(max_target_length):
            decoder_output, decoder_hidden = \
                self.decoder(decoder_input, decoder_hidden, encoder_outputs,
                        is_train = is_train)
            all_decoder_outputs[t] = decoder_output
            # select real target or decoder output
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = target_var[t]
            else:
                topv, topi = decoder_output.topk(1, dim=1)
                decoder_input = topi.squeeze(1)

        return all_decoder_outputs

    def response(self, input_var):
        # input_var size (length, 1)
        length = input_var.size(0)
        input_group = (input_var, [length])
        # outputs size (max_length, output_size)
        decoder_outputs = self.forward(input_group, teacher_forcing_ratio=0,
                is_train = False)
        return decoder_outputs

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=True):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout)

    def forward(self, inputs_seqs, input_lens, hidden=None):
        # embedded size (max_len, batch_size, hidden_size)
        embedded = self.embedding(inputs_seqs)
        packed = pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(outputs)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seqs, last_hidden, encoder_ouputs, is_train = True):
        # input_seqs size (batch_size,)
        # last_hidden size (n_layers, batch_size, hidden_size)
        # encoder_ouputs size (max_len, batch_size, hidden_size)
        batch_size = input_seqs.size(0)
        # embedded size (1, batch_size, hidden_size)
        embedded = self.embedding(input_seqs).unsqueeze(0)
        rnn_output, hidden = self.rnn(embedded, last_hidden)
        output = self.out(rnn_output)
        output = output.squeeze(0)
        return output, hidden
