import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from go_probe.utils.io import save_ckpt, load_ckpt

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

        losses = np.zeros((len(self.probes), self.label_dim))
        for i in range(len(self.probes)):
            for lbl in range(self.label_dim):
                loss = self.criterion(torch.from_numpy(preds[i][:, lbl]).float(),
                    torch.from_numpy(labels[:, lbl]).float())
                losses[i, lbl] = loss 

        def f1_metric(fpr, tpr, pos, neg):
            if tpr == 0:
                return 0
            recall = tpr
            precision = tpr * pos / (tpr * pos + fpr * neg)
            return 2 * precision * recall / (precision + recall)

        aucs = np.zeros((len(self.probes), self.label_dim))
        f1s = np.zeros((len(self.probes), self.label_dim))

        for i in range(len(self.probes)):
            for lbl in range(self.label_dim):
                labels_ = labels[:, lbl]
                preds_ = preds[i][:, lbl]
                auc = roc_auc_score(labels_, preds_)
                aucs[i, lbl] = auc

                pos = labels_.sum()
                neg = len(labels_) - pos 
                fprs, tprs, _ = roc_curve(labels_, preds_)
                f1 = max(f1_metric(fpr, tpr, pos, neg) for fpr, tpr in zip(fprs, tprs))
                f1s[i, lbl] = f1

        return {"loss": losses, "auc": aucs, "f1": f1s}

    def run(self):
        self.init_probes()
        best_metrics = {"loss": np.ones((len(self.probes), self.label_dim))*100,
                        "auc": np.zeros((len(self.probes), self.label_dim)),
                        "f1": np.zeros((len(self.probes), self.label_dim))}
                        
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

            metrics = self.report_metrics()
            best_metrics["loss"] = np.minimum(best_metrics["loss"], metrics["loss"])
            best_metrics["auc"] = np.maximum(best_metrics["auc"], metrics["auc"])
            best_metrics["f1"] = np.maximum(best_metrics["f1"], metrics["f1"])
            print(metrics["loss"].mean(axis = 1))
            print(metrics["auc"].mean(axis = 1))
            print(metrics["f1"].mean(axis = 1))
        return best_metrics
            