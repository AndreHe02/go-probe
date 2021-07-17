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

def probe_layers_parallel(train_dataset, test_dataset, get_representations, probes, optimizers, criterion, num_epochs, log_dir, batch_size=512):

    #logger = SummaryWriter(log_dir)
    best_test_losses = np.ones(len(probes)) * 1e10

    for epoch in range(num_epochs):
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

        train_counter = 0
        for X, y in tqdm(train_loader):
            layer_outputs = get_representations(X)
            y = y.type(torch.FloatTensor).cuda()
            for i in range(len(layer_outputs)):
                layer_outputs[i] = layer_outputs[i].cuda()

            for i, layer_output in enumerate(layer_outputs):
                pred = probes[i](layer_output)
                loss = criterion(pred, y)
                loss.backward(retain_graph=True)
                optimizers[i].step()
                optimizers[i].zero_grad()
                #logger.add_scalar('loss/train_layer_%d' % i, loss.data.item(), train_counter)
            train_counter += 1

        test_losses = np.zeros(len(probes))
        for X, y in tqdm(test_loader):
            layer_outputs = get_representations(X)
            y = y.type(torch.FloatTensor).cuda()
            for i in range(len(layer_outputs)):
                layer_outputs[i] = layer_outputs[i].cuda()

            for i, layer_output in enumerate(layer_outputs):
                pred = probes[i](layer_output)
                loss = criterion(pred, y).data.item()
                test_losses[i] += loss
        test_losses /= len(test_loader)
        print(test_losses)

        for i, test_loss in enumerate(test_losses):
            #logger.add_scalar('loss/test_layer_%d' % i, test_loss, epoch)
            if test_loss < best_test_losses[i]:
                best_test_losses[i] = test_loss
                save_checkpoint({
                    'epoch':epoch,
                    'state_dict':probes[i].state_dict(),
                    'optimizer':optimizers[i].state_dict()
                    }, os.path.join(log_dir, 'layer_%d.ckpt' % i))

    for i in range(len(probes)):
        load_from_ckpt(probes[i], os.path.join(log_dir, 'layer_%d.ckpt' % i))

def evaluate_auc(dataset, get_representations, probes):
    preds, targets = [[] for _ in range(len(probes))], []
    loader = DataLoader(dataset, shuffle=False, batch_size=512)
    for X, y in loader:
        layer_outputs = get_representations(X)
        y = y.type(torch.FloatTensor).cuda()
        for i in range(len(layer_outputs)):
            layer_outputs[i] = layer_outputs[i].cuda()

        for i, layer_output in enumerate(layer_outputs):
            pred = probes[i](layer_output)
            preds[i].append(pred.detach().cpu().numpy())
        targets.append(y.cpu().numpy())
    for i in range(len(probes)):
        preds[i] = np.concatenate(preds[i], axis=0)
    targets = np.concatenate(targets, axis=0)
    aucs = pd.DataFrame()
    for i in range(len(probes)):
        for kw in range(preds[i].shape[-1]):
            auc = roc_auc_score(targets[:,kw], preds[i][:,kw])
            fprs, tprs, _ = roc_curve(targets[:,kw], preds[i][:,kw])
            s = pd.Series({'layer':i, 'kw':kw, 'auc':auc, 'fprs_tprs':(fprs, tprs)})
            aucs = aucs.append(s, ignore_index=True)
    return aucs

def handcrafted_ft_probe_layers_parallel(dataset, prepare_batch, probes, optimizers, criterion, num_epochs, log_dir, num_workers=0):

    #logger = SummaryWriter(log_dir)
    best_test_losses = np.ones(len(probes)) * 1e10

    for epoch in range(num_epochs):
        dataset.shuffle('train')
        train_loader = dataset.loader('train', max_ram_files=50, num_workers=num_workers)
        test_loader = dataset.loader('test', max_ram_files=50)

        train_counter = 0
        for X, y in tqdm(train_loader):
            layer_outputs, y = prepare_batch(X, y)
            for i, layer_output in enumerate(layer_outputs):
                pred = probes[i](layer_output)
                loss = criterion(pred, y)
                loss.backward(retain_graph=True)
                optimizers[i].step()
                optimizers[i].zero_grad()
                #logger.add_scalar('loss/train_layer_%d' % i, loss.data.item(), train_counter)
            train_counter += 1

        test_losses = np.zeros(len(probes))
        for X, y in tqdm(test_loader):
            layer_outputs, y = prepare_batch(X, y)
            for i, layer_output in enumerate(layer_outputs):
                pred = probes[i](layer_output)
                loss = criterion(pred, y).data.item()
                test_losses[i] += loss
        test_losses /= len(test_loader)
        print(test_losses)

        for i, test_loss in enumerate(test_losses):
            #logger.add_scalar('loss/test_layer_%d' % i, test_loss, epoch)
            if test_loss < best_test_losses[i]:
                best_test_losses[i] = test_loss
                save_checkpoint({
                    'epoch':epoch,
                    'state_dict':probes[i].state_dict(),
                    'optimizer':optimizers[i].state_dict()
                    }, os.path.join(log_dir, 'layer_%d.ckpt' % i))

    for i in range(len(probes)):
        load_from_ckpt(probes[i], os.path.join(log_dir, 'layer_%d.ckpt' % i))

def handcrafted_ft_evaluate_loss(dataset, prepare_batch, probes, criterion):
    loader = dataset.loader('test', max_ram_files=25)

    test_losses = np.zeros(len(probes))
    for X, y in tqdm(test_loader):
        layer_outputs, y = prepare_batch(X, y)
        for i, layer_output in enumerate(layer_outputs):
            pred = probes[i](layer_output)
            loss = criterion(pred, y).data.item()
            test_losses[i] += loss
    test_losses /= len(test_loader)
    return test_losses

def handcrafted_ft_evaluate_auc(dataset, prepare_batch, probes):
    preds, targets = [[] for _ in range(len(probes))], []
    loader = dataset.loader('test', max_ram_files=25)
    for X, y in loader:
        layer_outputs, y = prepare_batch(X, y)
        for i, layer_output in enumerate(layer_outputs):
            pred = probes[i](layer_output)
            preds[i].append(pred.detach().cpu().numpy())
        targets.append(y.cpu().numpy())
    for i in range(len(probes)):
        preds[i] = np.concatenate(preds[i], axis=0)
    targets = np.concatenate(targets, axis=0)
    aucs = pd.DataFrame()
    for i in range(len(probes)):
        auc = roc_auc_score(targets.flatten(), preds[i].flatten())
        s = pd.Series({'layer':i, 'auc':auc})
        aucs = aucs.append(s, ignore_index=True)
    return aucs
