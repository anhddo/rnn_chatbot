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



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, decoder_ht, encoder_outputs):
        #decoder_hidden [1, bs, hidden_size]
        #encoder_outputs [max_length, bs, hidden_size]
        temp = self.attn(encoder_outputs)
        #[bs,1, hidden_size]x[bs, hidden_size, max_length]-> [bs,1, max_length]
        scores = torch.bmm(decoder_ht.transpose(0,1), temp.permute(1,2,0))
        attention_weights = F.softmax(scores, dim = 2)
        #[bs,1, max_length]x[bs, max_length, hidden_size]-> [bs,1, hidden_size]
        context = torch.bmm(attention_weights, encoder_outputs.permute(1,0,2))
        # context [1,bs, hidden_size]
        context = context.transpose(0,1)
        # concat_context_hidden [1,bs, hidden_size * 2]
        concat_context_hidden = torch.cat((context, decoder_ht), dim = 2)
        # output [1,bs, hidden_size]
        output = F.tanh(self.linear_out(concat_context_hidden))
        return output

            
