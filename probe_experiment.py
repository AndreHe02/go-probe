from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import os
import numpy as np

def train_epoch(feat_model, probe_model, loader, criterion, optimizer, logger, epoch):
    probe_model.train()
    for step, (board_repr, label) in enumerate(loader):
        board_repr = board_repr.type(torch.FloatTensor)
        board_repr = board_repr.cuda()
        label = label.cuda()
        feat = feat_model(board_repr).detach()
        pred = probe_model(feat)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if logger is not None:
            logger.add_scalar('loss/train', loss.data.item(), step+epoch*len(loader))

def test_epoch(feat_model, probe_model, loader, criterion, logger, epoch):
    probe_model.eval()
    test_set_loss = 0
    for step, (board_repr, label) in enumerate(loader):
        board_repr = board_repr.type(torch.FloatTensor)
        board_repr = board_repr.cuda()
        label = label.cuda()
        feat = feat_model(board_repr).detach()
        pred = probe_model(feat)
        loss = criterion(pred, label)
        test_set_loss += loss.data.item()
    avg_loss = test_set_loss / len(loader)
    if logger is not None:
        logger.add_scalar('loss/test', avg_loss, epoch)
    return avg_loss

def save_checkpoint(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)

class ProbeExperiment:

    def __init__(self, train_dataset, test_dataset, keywords):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.keywords = keywords

    def run(self, trial_name, feat_model, probe_model, configs):

        log_dir = os.path.join('experiments', trial_name)
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir) if configs['write_log'] else None

        train_loader = DataLoader(self.train_dataset, batch_size=configs['batch_size'], shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=configs['batch_size'], shuffle=False)

        criterion = configs['criterion']
        optimizer = configs['optimizer']

        best_loss = 1e10
        for epoch in range(configs['num_epochs']):

            train_epoch(feat_model, probe_model, train_loader, criterion, optimizer, logger, epoch)
            loss = test_epoch(feat_model, probe_model, test_loader, criterion, logger, epoch)

            if loss < best_loss:
                best_loss = loss
                print('[LOG] epoch %d loss %f, new best' % (epoch, loss))
                save_checkpoint({
                    'epoch':epoch,
                    'state_dict':probe_model.state_dict(),
                    'optimizer':optimizer.state_dict()
                    }, os.path.join(log_dir, 'best.ckpt'))
            else:
                print('[LOG] epoch %d loss %f' % (epoch, loss))
            if configs['save_ckpt']:
                save_checkpoint({
                    'epoch':epoch,
                    'state_dict':probe_model.state_dict(),
                    'optimizer':optimizer.state_dict()
                    }, os.path.join(log_dir, 'epoch%d.ckpt' % epoch))

        best_ckpt = torch.load(os.path.join(log_dir, 'best.ckpt'))
        probe_model.load_state_dict(best_ckpt['state_dict'])

    def predict_labels(self, feat_model, probe_model, dataset, batch_size=512):
        loader = DataLoader(dataset, batch_size, shuffle=False)
        probe_model.eval()
        feat_model.eval()
        pred_ls, label_ls = [], []
        for step, (board_repr, label) in enumerate(loader):
            board_repr = board_repr.type(torch.FloatTensor)
            board_repr = board_repr.cuda()
            feat = feat_model(board_repr).detach()
            pred = probe_model(feat)
            pred_ls.append(pred.detach().cpu().numpy())
            label_ls.append(label.detach().numpy())
        preds = np.concatenate(pred_ls, axis=0)
        labels = np.concatenate(label_ls, axis=0)
        return preds, labels
