# coding=utf8
import random
import json
import re
import data_utils
import argparse
import config


def load_conversations(file_path):
    dialogue_list = []
    with open(file_path) as file:
        for line in file:
            sp = line[:-1].split(' +++$+++ ')
            ua, ub, m = sp[:3]
            line_list = eval(sp[3])
            dialogue_list.append(line_list)
    return dialogue_list

def load_movie_lines(file_path):
    id2sentence = {}
    with open(file_path, encoding = 'iso-8859-1') as file:
        for line in file:
            sp = line[:-1].split(' +++$+++ ')
            lid, sentence = sp[0], sp[4]
            sentence = re.sub(r'[\.]+', '.', sentence)
            id2sentence[lid] = sentence
    return id2sentence

def augment_sentence(questions, answers, augment_q, augment_a):
    for pair in zip(questions, answers):
        Q = pair[0]
        A = pair[1]
        q_sentences = Q.strip('.').split('.')
        a_sentences = A.strip('.').split('.')
        for i in range(len(q_sentences)):
            for j in range(len(a_sentences)):
                augment_q.append('.'.join(q_sentences[i:]))
                augment_a.append('.'.join(a_sentences[:j+1]))

def augment_extract(dialogues, id2sentence):
    questions, answers = [], []
    augment_q, augment_a = [], []
    for ids in dialogues:
        join_func = lambda id: ' '.join(data_utils.basic_tokenizer(id2sentence[id]))
        sentences = [join_func(id) for id in ids]
        augment_sentence(sentences[:-1], sentences[1:], augment_q, augment_a)
        assert len(augment_q) == len(augment_a)
    return augment_q, augment_a

def normal_extract(dialogues, id2sentence):
    questions, answers = [], []
    for ids in dialogues:
        join_func = lambda id: ' '.join(data_utils.basic_tokenizer(id2sentence[id]))
        sentences = [join_func(id) for id in ids]
        questions.extend(sentences[:-1])
        answers.extend(sentences[1:])
    return questions, answers

def export_dialogue_corpus():
    dialogues = load_conversations(config.data_path + config.movie_conversations)
    id2sentence = load_movie_lines(config.data_path + config.movie_lines)
    if config.is_augment:
        print('augment data')
        questions, answers = augment_extract(dialogues, id2sentence)
    else:
        print('no augment')
        questions, answers = normal_extract(dialogues, id2sentence)
    dialogue_groups = zip(questions, answers)

    print('Dialogue pairs: %d' % len(questions))

    # random.shuffle(dialogue_corpus)
    with open(config.data_path + config.dialogue_corpus, 'w') as file:
        for a, b in dialogue_groups:
            file.write('%s +++$+++ %s\n' % (a, b))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-a', dest='augment', type=int)
    args = parser.parse_args()
    config.is_augment = args.augment
    config.parse('default')
    export_dialogue_corpus()
