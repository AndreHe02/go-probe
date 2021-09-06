import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

class BaseExperiment:

    def init_probes(self):
        return NotImplemented

    def dataloader(self, split):
        return NotImplemented
    
    def get_internal_reps(self, X):
        return NotImplemented

    def report_metrics(self):
        return NotImplemented
    
    def run(self):
        return NotImplemented

class DefaultExperiment(BaseExperiment):

    probe_dims = None
    label_dim = None
    num_epochs = None
    criterion = nn.BCEWithLogitsLoss()    
    
    def __init__(self, dataset):
        self.dataset = dataset

    def init_probes(self):
        self.probes = [nn.Linear(dim, self.label_dim).cuda() for dim in self.probe_dims]
        self.optims = [optim.Adam(probe.parameters(), lr=0.001) for probe in self.probes]
    
    def dataloader(self, split):
        if split == 'train':
            self.dataset.shuffle(split)
        return self.dataset.loader(split, max_ram_files=10)

    def report_metrics(self):
        test_loader = self.dataloader('test')
        preds = [[] for _ in range(len(self.probes))]
        labels = []
        for X, y in tqdm(test_loader):
            X = X.float().cuda()
            reps = self.get_internal_reps(X)
            batch_preds = [probe(rep) for probe, rep in zip(self.probes, reps)]
            for i, p in enumerate(batch_preds):
                preds[i].append(p.detach().cpu().numpy())
            labels.append(y.cpu().numpy())
        preds = [np.concatenate(p, axis=0) for p in preds]
        labels = np.concatenate(labels, axis=0)
        
        aucs = np.zeros((len(self.probes), self.label_dim))
        for i in range(len(self.probes)):
            for lbl in range(self.label_dim):
                labels_ = labels[:, lbl]
                preds_ = preds[i][:, lbl]
                auc = roc_auc_score(labels_, preds_)
                aucs[i, lbl] = auc
        return aucs

    def run(self):
        self.init_probes()
        best_aucs = np.zeros((len(self.probes), self.label_dim))
    
        for epoch in range(self.num_epochs):
            train_loader = self.dataloader('train')
            for X, y in tqdm(train_loader):
                X = X.float().cuda()
                y = y.float().cuda()
                reps = self.get_internal_reps(X)
                for rep, probe, optim in zip(reps, self.probes, self.optims):
                    pred = probe(rep)
                    loss = self.criterion(pred, y)
                    loss.backward(retain_graph=True)
                    optim.step()
                    optim.zero_grad()

            aucs = self.report_metrics()
            best_aucs = np.maximum(best_aucs, aucs)
            print(best_aucs.mean(axis = 1))
        return best_aucs
            