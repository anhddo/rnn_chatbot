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
import config
from model.attention import Attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, max_length = 20):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length

        for param in self.parameters():
            param.data.uniform_(-0.08, 0.08)

    def forward(self, input_group, target_group=(None, None),
            teacher_forcing_ratio=0.5, is_train = True):
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
        all_decoder_outputs = Variable(torch.zeros(max_target_length,\
                batch_size, self.decoder.output_size))
        # first decoder input
        decoder_input = Variable(torch.LongTensor([GO_token] * batch_size))

        if config.use_cuda:
            all_decoder_outputs.data = all_decoder_outputs.data.cuda()
            decoder_input.data = decoder_input.data.cuda()

        if config.encoder_bidirectional:
            encoder_ht, encoder_ct = encoder_hidden
            n_layers = encoder_ht.size(0)#n_layer*n_direction
            decoder_ht = torch.cat([encoder_ht[0:n_layers:2], encoder_ht[1:n_layers:2]], 2)
            decoder_ct = torch.cat([encoder_ct[0:n_layers:2], encoder_ct[1:n_layers:2]], 2)
            decoder_hidden = (decoder_ht, decoder_ct)
        else:
            decoder_hidden = encoder_hidden
        for t in range(max_target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input,\
                    decoder_hidden, encoder_outputs, is_train = is_train)
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
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, \
            embedding = None):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        if isinstance(embedding, nn.Embedding):
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(input_size, hidden_size)

        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers,\
                dropout=dropout, bidirectional = config.encoder_bidirectional)

    def forward(self, inputs_seqs, input_lens, hidden=None):
        # embedded size (max_len, batch_size, hidden_size)
        embedded = self.embedding(inputs_seqs)
        packed = pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(outputs)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1,\
            embedding = None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        if isinstance(embedding, nn.Embedding):
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(output_size, hidden_size)

        if config.use_attn:
            self.attention = Attention(self.hidden_size)

        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers,
                dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seqs, last_hidden, encoder_outputs, is_train = True):
        # input_seqs size (batch_size,)
        # last_hidden size (n_layers, batch_size, hidden_size)
        # encoder_ouputs size (max_len, batch_size, hidden_size)
        batch_size = input_seqs.size(0)
        # embedded size (1, batch_size, hidden_size)
        embedded = self.embedding(input_seqs).unsqueeze(0)
        # rnn_output,last_ht size (1, batch_size, hidden_size)
        output, hidden = self.rnn(embedded, last_hidden)
        if config.use_attn:
            output = self.attention(output, encoder_outputs)
        output = self.out(output)
        output = output.squeeze(0)
        return output, hidden
