# coding=utf8
import unicodedata
import json
import re
import random
import torch
from torch.autograd import Variable
from custom_token import *
import config
import operator


# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z',.!?]+", r" ", sentence)
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]

def build_DataLoader():
    pairs = []
    length_range = range(config.min_length, config.max_length)
    print('Loading Corpus.')
    with open(config.data_path + config.dialogue_corpus) as file:
        i = 0
        for line in file:
            i += 1
            pa, pb = line[:-1].split(' +++$+++ ')
            pa = pa.split()
            pb = pb.split()
            if len(pa) in length_range and len(pb) in length_range:
                pairs.append((pa, pb))

    print('Read dialogue pair: %d' % len(pairs))
    vocab = Vocabulary()
    for pa, pb in pairs:
        for word in pa + pb:
            vocab.index_word(word)
    vocab.trim()

    print('total pairs: %d' % (len(pairs)))
    loader = DataLoader(vocab, pairs)
    print('Batch number: %d' % len(loader))
    return loader

class Vocabulary(object):
    def __init__(self):
        self.trimmed = False
        self.reset()

    def reset(self):
        self.word2count = {}
        self.word2index = {"PAD": 0, "GO": 1, "EOS": 2, "UNK": 3}
        self.index2word = {0: "PAD", 1: "GO", 2: "EOS", 3: "UNK"}
        self.n_words = 4


    def index_word(self, word):
        if word not in self.word2count:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def trim(self):
        if self.trimmed:
            return
        self.trimmed = True

        # sortword2count
        self.word2count = sorted(self.word2count.items(), key = operator.itemgetter(1), reverse = True)
        print('total words %d' % (len(self.word2count)))
        self.word2count = self.word2count[:config.vocabulary_size]
        keep_words = [pair[0] for pair in self.word2count]
        self.reset()
        for word in keep_words:
            self.index_word(word)


class DataLoader(object):
    def __init__(self, vocabulary, src_pairs):
        batch_size = config.batch_size
        self.vocabulary = vocabulary
        self.data = []
        self.test = []

        n_iter = int(len(src_pairs)/ batch_size)
        n_test = int(n_iter * 0.1)
        assert(n_iter >= 10)

        src_pairs = sorted(src_pairs, key=lambda s: len(s[1]))

        train_decode_length, test_decoder_lenth = 0.0, 0.0
        # choose items to put into testset
        test_ids = random.sample(range(n_iter), n_test)
        for i in range(n_iter):
            batch_seq_pairs = sorted(src_pairs[i*batch_size: (i+1)*batch_size], key=lambda s: len(s[0]), reverse=True)
            decode_length = float(sum([len(x[1]) for x in batch_seq_pairs])) / 64

            input_group, target_group = self.__process(batch_seq_pairs)
            if i not in test_ids:
                self.data.append((input_group, target_group))
                train_decode_length += decode_length
            else:
                # test data
                self.test.append((input_group, target_group))
                test_decoder_lenth += decode_length

        self.train_data_len = len(self.data)
        self.test_data_len = len(self.test)
        mean_train_decode_len = train_decode_length / self.train_data_len
        mean_test_decode_len = test_decoder_lenth / self.test_data_len
        print('mean decode length: (%.2f, %.2f)' % (mean_train_decode_len, mean_test_decode_len))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_vocabulary_size(self):
        return self.vocabulary.n_words

    def random_batch(self):
        # assert(self.train_data_len > 1)
        return self.data[random.randint(0, self.train_data_len-1)]

    def random_test(self):
        # assert(self.test_data_len > 1)
        return self.test[random.randint(0, self.test_data_len-1)]

    def __process(self, batch_seq_pairs):
        input_seqs, target_seqs = zip(*batch_seq_pairs)
        # convert to index
        input_seqs = [self.indexes_from_sentence(s) for s in input_seqs]

        if config.reverse_input:
            input_seqs = [s[::-1] for s in input_seqs]

        target_seqs = [self.indexes_from_sentence(s) for s in target_seqs]
        # PAD input_seqs
        input_lens = [len(s) for s in input_seqs]
        max_input_len = max(input_lens)
        input_padded = [self.pad_seq(s, max_input_len) for s in input_seqs]
        # PAD target_seqs
        target_lens = [len(s) for s in target_seqs]
        max_target_len = max(target_lens)
        target_padded = [self.pad_seq(s, max_target_len) for s in target_seqs]
        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
        target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
        if config.use_cuda:
            input_var = input_var.cuda()
            target_var = target_var.cuda()
        return (input_var, input_lens), (target_var, target_lens)

    def indexes_from_sentence(self, sentence):
        to_index = lambda word: self.vocabulary.word2index[word] if word in \
                self.vocabulary.word2index else UNK_token
        return [to_index(word) for word in sentence] + [EOS_token]

    def pad_seq(self, seq, max_length):
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq
