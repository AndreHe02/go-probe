from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, roc_curve
import os
import pandas as pd

def save_checkpoint(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)

def load_from_ckpt(model, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])

class BaseExperiment:
    log_dir = None
    criterion = None
    num_epochs = None
    batch_size = None

    @classmethod
    def init_probes(cls):
        return NotImplemented

    @classmethod
    def init_optimizers(cls):
        return NotImplemented

    @classmethod
    def dataloader(cls, split):
        return NotImplemented

    @classmethod
    def board_to_features(cls, X):
        return NotImplemented

    @classmethod
    def report_metrics(cls, probes):
        return NotImplemented

class DefaultExperiment(BaseExperiment):
    criterion = torch.nn.BCEWithLogitsLoss()
    num_epochs = 5
    probe_dims = None

    


def run(e):
    probes = e.init_probes()
    optimizers = e.init_optimizers()
    min_losses = np.ones(len(probes)) * 1e10
    step_counter = 0
    for epoch in range(e.num_epochs):
        train_loader = e.dataloader('train')
        test_loader = e.dataloader('test')
        for X, y in tqdm(train_loader):
            features = e.board_to_features(X)
            for i, (ft, probe, optim) in enumerate(zip(features, probes, optimizers)):
                pred = probe(ft)
                loss = e.criterion(pred, y)
                loss.backward(retrain_graph=True)
                optim.step()
                optim.zero_grad()
                e.logger.add_scalar('loss/train_ft_%d'%i, loss.data.item(), step_counter)
                step_counter += 1
        test_losses = np.zeros(len(probes))
        test_batches = 0
        for X, y in test_loader:
            features = e.board_to_features(X)
            for i, (ft, probe) in enumerate(zip(features, probes)):
                pred = probe(ft)
                loss = e.criterion(pred, y).data.item()
                test_losses[i] += loss
            test_batches += 1
        test_losses / test_batches
        



