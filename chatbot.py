# coding=utf8
import json
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from model_utils import load_vocabulary, build_model, BotAgent
import argparse
import config


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c', dest='config')
    parser.add_argument('-f', dest='test_file')
    parser.add_argument('-e', dest='epoch', type = int)
    args = parser.parse_args()
    config.parse(args.config)
    vocab = load_vocabulary()
    model = build_model(len(vocab.word2index), load_ckpt=True,
            ckpt_epoch=args.epoch)
    config.use_cuda = False
    model.cpu()
    bot = BotAgent(model, vocab)
    if args.test_file is not None:
        with open(args.test_file) as file:
            question_list = []
            for line in file:
                question_list.append(line[:-1])
            for question in question_list:
                print('> %s' % question)
                print('bot: %s' % bot.response(question))
    else:
        while True:
            user_input = input('me: ')
            if user_input.strip() == '':
                continue
            print('%s: %s' % ('bot', bot.response(user_input)))

if __name__ == '__main__':
    main()
