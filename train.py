# coding=utf8
import os
import json
import math
import time
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm
from data_utils import build_DataLoader
from masked_cross_entropy import *
from model_utils import build_model, save_model, model_evaluate,\
        save_vocabulary, get_ckpts

with open('config.json') as config_file:
    config = json.load(config_file)
USE_CUDA = config['TRAIN']['CUDA']

n_iters = config['TRAIN']['N_EPOCHS']
batch_size = config['TRAIN']['BATCH_SIZE']
clip = config['TRAIN']['CLIP']
learning_rate = config['TRAIN']['LEARNING_RATE']
teacher_forcing_ratio = config['TRAIN']['TEACHER_FORCING_RATIO']

print_every = 200
save_every = print_every * 10

def main():
    dataset = build_DataLoader(batch_size=batch_size)
    vocabulary_list = sorted(dataset.vocabulary.word2index.items(), key=lambda x: x[1])
    save_vocabulary(vocabulary_list)
    vocab_size = dataset.get_vocabulary_size()
    model = build_model(vocab_size, load_ckpt = True)
    model_optimizer = optim.SGD(model.parameters(), lr = learning_rate)


    start = time.time()
    dataset_len = len(dataset)
    ckpts = get_ckpts()
    iter_idx = 0 if len(ckpts) == 0 else max(ckpts)
    print_loss_total = 0.0
    # scheduler = optim.lr_scheduler.MultiStepLR(model_optimizer,
    #         milestones=[30,80, 1000], gamma=0.1, last_epoch = iter_idx)
    print('Start Training.')
    while iter_idx < n_iters:
        iter_idx += 1
        input_group, target_group = dataset.random_batch()
        # zero gradients
        model_optimizer.zero_grad()
        # run seq2seq
        all_decoder_outputs = model(input_group, target_group, teacher_forcing_ratio=1)
        target_var, target_lens = target_group
        # loss calculation and backpropagation
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),
            target_var.transpose(0, 1).contiguous(),
            target_lens
        )
        print_loss_total += loss.data[0]
        loss.backward()
        clip_grad_norm(model.parameters(), clip)
        # update parameters
        model_optimizer.step()

        if iter_idx % print_every == 0:
            test_loss = model_evaluate(model, dataset)
            print_summary(start, iter_idx, math.exp(print_loss_total / print_every))
            print('Test PPL: %.4f, lr=%f' % (math.exp(test_loss),
                model_optimizer.param_groups[0]['lr']))
            print_loss_total = 0.0
            hot_update_lr(model_optimizer)
            if iter_idx % save_every == 0:
                save_model(model, iter_idx)
        # break
    save_model(model, iter_idx)

def hot_update_lr(model_optimizer):
    with open('config.json') as config_file:
        config = json.load(config_file)
    learning_rate = config['TRAIN']['LEARNING_RATE']
    for param_group in model_optimizer.param_groups:
            param_group['lr'] = learning_rate


def print_summary(start, epoch, print_ppl_avg):
    output_log = '%s (epoch: %d finish: %d%%) PPL: %.4f' %\
        (time_since(start, float(epoch) / n_iters), epoch, float(epoch) /
                n_iters * 100, print_ppl_avg)
    print(output_log)

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


if __name__ == '__main__':
    main()
