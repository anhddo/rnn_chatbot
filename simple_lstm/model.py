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

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, max_length=20, tie_weights=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length

        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

        for param in self.parameters():
            param.data.uniform_(-0.08, 0.08)

        if tie_weights:
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def forward(self, input_group, target_group=(None, None), teacher_forcing_ratio=0.5):
        input_var, input_lens = input_group
        encoder_outputs, encoder_hidden = self.encoder(input_var, input_lens)

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
        if USE_CUDA:
            all_decoder_outputs = all_decoder_outputs.cuda()
            decoder_input = decoder_input.cuda()
        for t in range(max_target_length):
            decoder_output, decoder_hidden = \
                self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs[t] = decoder_output
            # select real target or decoder output
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = target_var[t]
            else:
                # decoder_output = F.log_softmax(decoder_output)
                topv, topi = decoder_output.data.topk(1, dim=1)
                decoder_input = Variable(topi.squeeze(1))

        return all_decoder_outputs

    def response(self, input_var):
        # input_var size (length, 1)
        length = input_var.size(0)
        input_group = (input_var, [length])
        # outputs size (max_length, output_size)
        decoder_outputs = self.forward(input_group, teacher_forcing_ratio=0)
        # topv, topi = decoder_outputs.data.topk(1, dim=1)
        # decoder_index = topi.squeeze(1)
        # return decoder_index
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
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers=n_layers, dropout=dropout)
        if USE_CUDA:
            self.rnn = self.rnn.cuda()

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
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers=n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        if USE_CUDA:
            self.rnn = self.gru.cuda()
            self.out = self.out.cuda()

    def forward(self, input_seqs, last_hidden, encoder_ouputs):
        # input_seqs size (batch_size,)
        # last_hidden size (n_layers, batch_size, hidden_size)
        # encoder_ouputs size (max_len, batch_size, hidden_size)
        batch_size = input_seqs.size(0)
        # embedded size (1, batch_size, hidden_size)
        embedded = self.embedding(input_seqs).unsqueeze(0)
        # output size (1, batch_size, hidden_size)
        output, hidden = self.rnn(embedded, last_hidden)
        # attn_weights size (batch_size, 1, max_len)
        output = self.out(concat_output)
        return output, hidden
