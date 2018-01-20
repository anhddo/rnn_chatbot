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
import numpy as np


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, max_length = 20):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length

        for param in self.parameters():
            param.data.uniform_(-0.08, 0.08)

    def forward(self, criterion, input_group, target_group=(None, None),
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

        decoder_hidden = tuple([x[:config.n_decoder_layers]for x in encoder_hidden])

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


        loss = 0
        if is_train:
            loss = self.calc_loss(criterion, target_group, all_decoder_outputs)
        return all_decoder_outputs, loss

    def calc_loss(self, criterion, target_group, all_decoder_outputs):
        target_var, target_lens = target_group
        #sort reverse order
        sort_index = np.argsort(target_lens)[::-1].tolist()
        #apply sort index to array
        sorted_target_lens = [target_lens[i] for i in sort_index]
        sort_index = torch.LongTensor(sort_index)
        if config.use_cuda:
            sort_index = sort_index.cuda()
        sorted_decoder_output = torch.index_select(all_decoder_outputs,
                1,Variable(sort_index))
        sorted_target_var = torch.index_select(target_var,  1, Variable(sort_index))
        #pack input and target to apply crossentropyloss
        input_loss = pack_padded_sequence(sorted_decoder_output,
                sorted_target_lens)
        target_loss = pack_padded_sequence(sorted_target_var,
                sorted_target_lens)
        loss = criterion(input_loss.data, target_loss.data)
        return loss


    def response(self, input_var):
        # input_var size (length, 1)
        length = input_var.size(0)
        input_group = (input_var, [length])
        # outputs size (max_length, output_size)
        decoder_outputs = self.forward(criterion = None,input_group =
                input_group, teacher_forcing_ratio=0,
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
        if config.encoder_bidirectional:
            outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1,\
            embedding = None):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        if isinstance(embedding, nn.Embedding):
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(output_size, hidden_size)

        if config.use_attn:
            self.attention = Attention(hidden_size)

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
