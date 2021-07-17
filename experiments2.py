import torch.nn as nn

class BaseExperiment:

    criterion = NotImplemented

    def __init__(self, dir):
        self.probes = self.initialize_probes()
        self.optimizers = self.initialize_optimizers()

    def initialize_probes(self):
        return NotImplemented

    def initialize_optimizers(self):
        return NotImplemented

    def loader(self, dataset):
        return NotImplemented

    #can put tqdm in loader
    def train(self, dataset):
        loader = self.loader(dataset)
        for X, y in loader:

            #dont put extra bullshit in forward. just write a existence dataset dumbass


    def _run(self, train_split, test_split, num_epochs):
        for epoch in range(self.num_epochs):
            self.train(train_split)
            self.callback(self.test(test_split))
        self.load()
        return NotImplemented

    def run(self, dataset, num_epochs):
        return NotImplemented


    def load(self):
        return NotImplemented

class LanguageProbeExperiment:
