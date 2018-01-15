# coding=utf8
import os
import sys
import json
import random
import torch
import data_utils
from data_utils import Vocabulary
from masked_cross_entropy import *
from custom_token import *
import numpy as np
import glob
from importlib import import_module
import config

question_list = []
with open('test_questions.txt') as file:
    for line in file:
        question_list.append(line[:-1])

def model_evaluate(model, dataset, evaluate_num=10, auto_test=True):
    model.train(False)
    total_loss = 0.0
    for _ in range(evaluate_num):
        input_group, target_group = dataset.random_test()
        all_decoder_outputs = model(input_group,
                target_group,
                teacher_forcing_ratio=1)
        target_var, target_lens = target_group
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),
            target_var.transpose(0, 1).contiguous(),
            target_lens
        )
        total_loss += loss.data[0]
        # format_output(dataset.vocabulary.index2word, input_group,\
        #target_group, all_decoder_outputs)
    if auto_test is True:
        bot = BotAgent(model, dataset.vocabulary)
        for question in question_list:
            print('> %s' % question)
            print('bot: %s' % bot.response(question))
    model.train(True)
    return total_loss / evaluate_num

def get_all_ckpts_file():
    ckpts = []
    file_names = glob.glob('%s%s*' % (config.checkpoint_path, config.prefix))
    return file_names

def convert_ckpt_str_to_array(file_names):
    ckpts = [int(name.split('_')[-1]) for name in file_names]
    return ckpts

def get_ckpts():
    file_names = get_all_ckpts_file()
    ckpts = convert_ckpt_str_to_array(file_names)
    return ckpts

def max_ckpts():
    ckpts = get_ckpts()
    retun -1 if len(ckpts) == 0 else max(ckpts)

def build_model(vocab_size, load_ckpt=False, ckpt_epoch = -1):
    model_pkg = import_module('model.%s'%(config.model_name))
    model = model_pkg.create_model(vocab_size)

    if load_ckpt is True and os.path.exists(config.checkpoint_path) is True:
        # load checkpoint
        prefix = config.prefix
        model_path = None
        if ckpt_epoch >= 0:
            model_path = '%s%s_%d' % (config.checkpoint_path, prefix, ckpt_epoch)
        else:
            # use last checkpoint
            ckpts = get_ckpts()
            if len(ckpts) > 0:
                model_path = '%s%s_%d' % (config.checkpoint_path, prefix, max(ckpts))

        print(model_path)
        if model_path is not None and os.path.exists(model_path):
            torch_object = torch.load(model_path)
            model.load_state_dict(torch_object)
            print('Load %s' % model_path)

    if config.use_cuda:
        model = model.cuda()
    return model

def init_path():
    if os.path.exists(config.checkpoint_path) is False:
        os.mkdir(config.checkpoint_path)

def save_model(model, iter_idx):
    init_path()
    file_names = get_all_ckpts_file()
    ckpts = convert_ckpt_str_to_array(file_names)
    sorted_idx = np.argsort(ckpts)
    if len(ckpts) >= 3:
        for idx in sorted_idx[:-2]:
            os.remove(file_names[idx])
    save_path = '%s%s_%d' % (config.checkpoint_path, config.prefix, iter_idx)
    torch.save(model.state_dict(), save_path)

def save_vocabulary(vocabulary_list):
    init_path()
    with open(config.vocabulary_path, 'w') as file:
        for word, index in vocabulary_list:
            file.write('%s %d\n' % (word, index))

def load_vocabulary():
    if os.path.exists(config.vocabulary_path):
        word2index = {}
        with open(config.vocabulary_path) as file:
            for line in file:
                line_spl = line[:-1].split()
                word2index[line_spl[0]] = int(line_spl[1])
        index2word = dict(zip(word2index.values(), word2index.keys()))
        vocab = Vocabulary()
        vocab.word2index = word2index
        vocab.index2word = index2word
        return vocab
    else:
        raise('not found %s' % config.vocabulary_path)

class BotAgent(object):
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    def response(self, question):
        input_var = self.build_input_var(question)
        if input_var is None:
            return "sorry, I don 't know ."
        decoder_output = self.model.response(input_var)
        decoder_output = decoder_output.squeeze(1)
        topv, topi = decoder_output.data.topk(1, dim=1)
        topi = topi.squeeze(1)
        if config.use_cuda:
            preidct_resp = topi.cpu().numpy()
        else:
            preidct_resp = topi.numpy()
        resp_words = self.build_sentence(preidct_resp)
        return resp_words

    def build_input_var(self, user_input):
        words = data_utils.basic_tokenizer(user_input)
        words_index = []
        unknown_words = []
        for word in words:
            if word in self.vocab.word2index.keys():
                # keep known words
                words_index.append(self.vocab.word2index[word])
            else:
                unknown_words.append(word)
        if len(unknown_words) > 0:
            print('unknown_words: ' + str(unknown_words))
        # append EOS token
        words_index.append(EOS_token)

        if config.reverse_input:
            words_index = words_index[::-1]

        if len(words_index) > 0:
            input_var = Variable(torch.LongTensor([words_index])).transpose(0, 1)
            if config.use_cuda:
                input_var = input_var.cuda()
            # input_var size (length, 1)
            return input_var
        return None

    def build_sentence(self, words_index):
        resp_words = []
        for index in words_index:
            if index < 3:
                # end sentence
                break
            resp_words.append(self.vocab.index2word[index])
        return ' '.join(resp_words)

if __name__ == '__main__':
    pass
