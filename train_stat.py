import config
import pickle
import os
import matplotlib.pyplot as plt

class TrainStat:
    def __init__(self):
        self.iters = []
        self.train_loss = []
        self.test_loss = []

    def add(self, iter_idx, train_loss, test_loss):
        self.iters.append(iter_idx)
        self.train_loss.append(train_loss)
        self.test_loss.append(test_loss)

    def plot(self):
        plt.plot(self.iters, self.train_loss, self.iters, self.test_loss)
        plt.savefig(config.checkpoint_path + 'loss.png')

    def load(self):
        if os.path.exists(config.train_stat_path):
            file_pi = open(config.train_stat_path , 'rb') 
            obj = pickle.load(file_pi) 
            self.__dict__.update(obj.__dict__)

    def save(self):
        file_pi = open(config.train_stat_path , 'wb') 
        pickle.dump(self, file_pi) 
