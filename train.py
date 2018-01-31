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
from model_utils import build_model, save_model, model_evaluate,\
        save_vocabulary, get_ckpts
import argparse
import config
import numpy as np
from train_stat import TrainStat


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c', dest='config')
    args = parser.parse_args()
    config.parse(args.config)
    train()

def init_milestone(total_batch):
    milestones = [total_batch * 5] * 10
    milestones[0] = total_batch * 50
    milestones[1] = total_batch * 15
    return np.cumsum(milestones)

def train():
    train_stat = TrainStat()
    train_stat.load()
    print(train_stat.iters)

    dataset = build_DataLoader()
    vocabulary_list = sorted(dataset.vocabulary.word2index.items(),\
            key=lambda x: x[1])
    save_vocabulary(vocabulary_list)
    vocab_size = dataset.get_vocabulary_size()
    model = build_model(vocab_size, load_ckpt = True)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr = config.learning_rate)

    criterion = nn.CrossEntropyLoss()
    start = time.time()
    total_batch = len(dataset)
    ckpts = get_ckpts()
    iter_idx = 0 if len(ckpts) == 0 else max(ckpts)
    print_loss_total = 0.0

    milestones = init_milestone(total_batch)
    n_iters = milestones[-1]

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
            milestones = milestones, gamma=0.5)
    scheduler.step(iter_idx)
    print('Start Training. total: %s iterations'%n_iters)
    while iter_idx < n_iters:
        iter_idx += 1
        scheduler.step()
        input_group, target_group = dataset.random_batch()
        # zero gradients
        optimizer.zero_grad()
        # run seq2seq
        _, loss = model(criterion, input_group, target_group, 
                teacher_forcing_ratio=1)
        print_loss_total += loss.data[0]
        loss.backward()
        clip_grad_norm(model.parameters(), config.clip)
        # update parameters
        optimizer.step()

        if iter_idx % config.print_every == 0:
            test_loss = model_evaluate(model, criterion, dataset)
            test_loss = math.exp(test_loss)
            train_loss = math.exp(print_loss_total / config.print_every)
            print_summary(start, iter_idx, n_iters, train_loss,\
                    optimizer.param_groups[0]['lr']) 
            print('Test loss: %.4f ' % (test_loss))
            print_loss_total = 0.0
            train_stat.add(iter_idx, train_loss, test_loss)
            train_stat.plot()
            # hot_update_lr(optimizer)
        if iter_idx % config.save_every == 0:
            save_model(model, iter_idx)
            train_stat.save()
    save_model(model, iter_idx)

def print_summary(start, epoch, n_iters, print_ppl_avg, lr):
    output_log = '%s (iter: %d finish: %d%%) loss: %.4f, lr=%f' %\
        (time_since(start, float(epoch) / n_iters), epoch, float(epoch) /
                n_iters * 100, print_ppl_avg, lr)
    print(output_log)
    with open(config.checkpoint_path+'log.txt', "a") as myfile:
        myfile.write(output_log+'\n')

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
