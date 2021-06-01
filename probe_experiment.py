from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import os
import numpy as np

log_dir = 'experiments'

def save_checkpoint(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)

def load_from_ckpt(model, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])

class ProbeExperiment:

    def __init__(self, train_loader, test_loader, feat_model, keywords):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.feat_model = feat_model
        self.keywords = keywords

    def train_epoch(self, probe_model, criterion, optimizer, logger, epoch):
        probe_model.train()
        self.feat_model.eval()
        for step, (board_repr, label) in enumerate(self.train_loader):
            board_repr = board_repr.type(torch.FloatTensor)
            board_repr = board_repr.cuda()
            label = label.cuda()
            feat = self.feat_model(board_repr).detach()
            pred = probe_model(feat)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if logger is not None:
                logger.add_scalar('loss/train', loss.data.item(),
                    step+epoch*len(self.train_loader))

    def test_epoch(self, probe_model, criterion, logger, epoch):
        probe_model.eval()
        self.feat_model.eval()
        cumulative_loss = 0
        for step, (board_repr, label) in enumerate(self.test_loader):
            board_repr = board_repr.type(torch.FloatTensor)
            board_repr = board_repr.cuda()
            label = label.cuda()
            feat = self.feat_model(board_repr).detach()
            pred = probe_model(feat)
            loss = criterion(pred, label)
            cumulative_loss += loss.data.item()
        avg_loss = cumulative_loss / len(self.test_loader)
        if logger is not None:
            logger.add_scalar('loss/test', avg_loss, epoch)
        return avg_loss

    def run(self, probe_model, criterion, optimizer, num_epochs, configs):
        trial_name = configs['name']
        write_log = configs['write_log']
        save_ckpt = configs['save_ckpt']
        trial_dir = os.path.join(log_dir, trial_name)
        os.makedirs(trial_dir, exist_ok=True)
        logger = SummaryWriter(trial_dir) if write_log else None

        best_loss = 1e10
        for epoch in range(num_epochs):
            self.train_epoch(probe_model, criterion, optimizer, logger, epoch)
            loss = self.test_epoch(probe_model, criterion, logger, epoch)
            if loss < best_loss:
                best_loss = loss
                print('[LOG] epoch %d loss %f, new best' % (epoch, loss))
                save_checkpoint({
                    'epoch':epoch,
                    'state_dict':probe_model.state_dict(),
                    'optimizer':optimizer.state_dict()
                    }, os.path.join(trial_dir, 'best.ckpt'))
            else:
                print('[LOG] epoch %d loss %f' % (epoch, loss))
            if save_ckpt:
                save_checkpoint({
                    'epoch':epoch,
                    'state_dict':probe_model.state_dict(),
                    'optimizer':optimizer.state_dict()
                    }, os.path.join(trial_dir, 'epoch%d.ckpt' % epoch))
        load_from_ckpt(probe_model, os.path.join(trial_dir, 'best.ckpt'))

    def get_test_predictions(self, probe_model, loader):
        '''must fit into memory'''
        probe_model.eval()
        self.feat_model.eval()
        pred_ls, label_ls = [], []
        for step, (board_repr, label) in enumerate(loader):
            board_repr = board_repr.type(torch.FloatTensor)
            board_repr = board_repr.cuda()
            feat = self.feat_model(board_repr).detach()
            pred = probe_model(feat)
            pred_ls.append(pred.detach().cpu().numpy())
            label_ls.append(label.detach().numpy())
        preds = np.concatenate(pred_ls, axis=0)
        labels = np.concatenate(label_ls, axis=0)
        return preds, labels
