from handcrafted_datasets import Dataset, CrossValDataset
from experiments import *
from models import *
import torch.nn as nn
import torch.optim as optim
from io_utils import *
from argparse import ArgumentParser

model = load_go_model_from_ckpt('model_ckpt.pth.tar', rm_prefix=True).cuda()
n_channels = [8, 64, 64, 64, 48, 48, 32, 32]
criterion = nn.BCEWithLogitsLoss()
num_epochs = 5

def probe_positional(ft):
    dataset = Dataset(ft, 'C:\\Users\\andre\\go-ai\\data', 0.8, 0.2, 1024)

    def positional_labels(X, y):
        X = X.type(torch.FloatTensor).cuda()
        layer_outputs = model.forward_layer_outputs(X)
        y = y.type(torch.FloatTensor).cuda()
        return layer_outputs, y.flatten(start_dim=1)

    probes = [nn.Sequential(nn.Conv2d(nc, 1, 19, padding=9), nn.Flatten()).cuda() for nc in n_channels]
    optimizers = [optim.Adam(probes[i].parameters()) for i in range(len(probes))]

    handcrafted_ft_probe_layers_parallel(dataset, positional_labels, probes, optimizers, criterion, num_epochs, 'temp', 0)
    return handcrafted_ft_evaluate_auc(dataset, positional_labels, probes)

def probe_existential(ft, dataset):

    def board_labels(X, y):
        X = X.type(torch.FloatTensor).cuda()
        layer_outputs = model.forward_layer_outputs(X)

        if len(y.shape) == 4:
            y = y[:, 0, :, :]

        if len(y.shape) > 1:
            y = torch.sum(y, dim=(1, 2))
            y = (y > 0)

        return layer_outputs, y.unsqueeze(dim=1).type(torch.FloatTensor).cuda()

    probes = [nn.Sequential(nn.Flatten(), nn.Linear(nc*19*19, 1)).cuda() for nc in n_channels]
    optimizers = [optim.Adam(probes[i].parameters()) for i in range(len(probes))]

    handcrafted_ft_probe_layers_parallel(dataset, board_labels, probes, optimizers, criterion, num_epochs, 'temp')
    return handcrafted_ft_evaluate_auc(dataset, board_labels, probes)

def get_mht_dist_loss():
    import numpy as np
    mht_dist_grid = np.zeros((39, 39))
    for dist in range(39):
        for i in range(dist+1):
            j = dist - i
            for r1, c1 in ((i, j), (i, -j), (-i, j), (-i, -j)):
                r, c = r1+19, c1+19
                if 0 <= r < 39 and 0 <= c < 39:
                    mht_dist_grid[r][c] = dist

    def mht_grid_19x19(center_r, center_c):
        return mht_dist_grid[19-center_r:38-center_r, 19-center_c:38-center_c]

    mht_grids = {pt:torch.Tensor(mht_grid_19x19(pt//19, pt%19)).detach() for pt in range(19*19)}

    def weighted_mht_dist_loss(pred, target):
        mht = [mht_grids[int(pt.data.item())] for pt in target]
        mht = torch.stack(mht)
        mht_flat = nn.Flatten()(mht).cuda()
        pred = nn.Softmax()(nn.Flatten()(pred))
        return torch.mean(torch.sum(pred * mht_flat, dim=1))

    return weighted_mht_dist_loss

#special case for the "recent move" feature
def probe_recent_move(ft, dataset):

    def board_labels(X, y):
        X = X.type(torch.FloatTensor).cuda()
        layer_outputs = model.forward_layer_outputs(X)
        return layer_outputs, y

    probes = [nn.Sequential(nn.Flatten(), nn.Linear(nc*19*19, 1)).cuda() for nc in n_channels]
    optimizers = [optim.Adam(probes[i].parameters()) for i in range(len(probes))]
    criterion = get_mht_dist_loss()

    handcrafted_ft_probe_layers_parallel(dataset, board_labels, probes, optimizers, criterion, num_epochs, 'temp')
    losses = handcrafted_ft_evaluate_loss(dataset, board_labels, probes, criterion)
    return losses

def main(args):
    n_fold = 10
    ft = args.feature

    cross_val_datasets = CrossValDataset(ft, 'C:\\Users\\andre\\go-ai\\data', 10, 1024)

    for val_set in range(n_fold):
        dataset = cross_val_datasets.train_val_split(val_set)
        aucs = probe_existential(ft, dataset)
        aucs['validation_set'] = val_set
        print(aucs)
        write_pkl(aucs, ft+'_val%d.pkl' % val_set)



    '''
    try:
        pos_aucs = probe_positional(ft)
        print(pos_aucs)
        write_pkl(pos_aucs, ft+'_pos.pkl')
    except:
        print('skipped')

    try:
        count_aucs = probe_count(ft, th)
        print(count_aucs)
        write_pkl(count_aucs, ft+'_count.pkl')
    except:
        print('skipped')
    '''

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--feature', '-f', type=str)
    args = parser.parse_args()
    main(args)
