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
from model.seq2seq import Seq2Seq, Encoder, Decoder

def create_model(vocab_size):
    embedding = nn.Embedding(vocab_size, config.hidden_size) \
            if config.single_embedding else None

    encoder = Encoder(vocab_size, config.hidden_size, \
            n_layers = config.n_encoder_layers, dropout=config.dropout)


    decoder = Decoder(config.hidden_size, vocab_size,\
            n_layers = config.n_decoder_layers, dropout=config.dropout)

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        max_length = config.max_length
    )

    if config.use_cuda:
        model.cuda()

    return model
