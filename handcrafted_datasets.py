import random
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import torch

from util import suppress_stdout


def collate(batch):
    progress, inputs, labels = batch[0]
    input_shape = inputs.shape
    return progress, torch.cat([torch.FloatTensor(inputs), torch.ones((input_shape[0], 1, input_shape[2], input_shape[3]))], dim=1), torch.LongTensor(labels)

class Dataset:
    def __init__(self, data_dir, train_files, batch_size, num_val_files=1000, num_test_files=1000, seed=0):
        random.seed(seed)
        self.batch_size = batch_size
        self.data_dir = data_dir
        for root, _, files in os.walk(data_dir):
            filenames = [os.path.join(root, fname) for fname in files if 'features' in fname]
        data_files = [(fname, fname.replace('features', 'labels')) for fname in filenames]
        random.shuffle(data_files)
        self.splits = {'val': data_files[:num_val_files], 'test': data_files[num_val_files:num_val_files+num_test_files], 'train': data_files[num_val_files+num_test_files:num_val_files+num_test_files+train_files]}
        print('done loading data')
        print('split sizes:')
        for key in ['train', 'val', 'test']:
            print(key, len(self.splits[key]))

    def shuffle(self, split, seed=None):
        assert split in ['train', 'val', 'test']
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.splits[split])


    def loader(self, split, max_ram_files=100, num_workers=20):
        assert split in ['train', 'val', 'test']
        return torch.utils.data.DataLoader(SplitLoader(self.splits[split], self.batch_size, max_ram_files), batch_size=1, pin_memory=True, collate_fn=collate, num_workers=num_workers) # just 1 worker since it should be super fast anyway


class SplitLoader(torch.utils.data.IterableDataset):
    def __init__(self, filenames, batch_size, max_ram_files=1):
        super(SplitLoader).__init__()
        self.filenames = filenames
        self.batch_size = batch_size
        self.max_ram_files = max_ram_files
        self.loaded_inputs, self.loaded_labels = [], []
        self.prev_load_pos = 0
        self.load_pos = 0
        self.pos = 0


    def __len__(self):
        return len(self.filenames)


    def __iter__(self):
        return self


    def __next__(self):
        increment = self.max_ram_files
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None: # # in a worker process
            increment = self.max_ram_files * worker_info.num_workers
            worker_id = worker_info.id
            if self.load_pos == 0:
                self.load_pos = worker_id * self.max_ram_files
        if self.pos >= len(self.loaded_labels):
            if self.load_pos >= len(self.filenames):
                raise StopIteration
            self.loaded_inputs, self.loaded_labels = [], []
            for i in range(self.load_pos, min(len(self.filenames), self.load_pos + self.max_ram_files)):
                features_file, labels_file = self.filenames[i]
                self.loaded_inputs.append(np.load(features_file))
                self.loaded_labels.append(np.load(labels_file))
            self.loaded_inputs = np.concatenate(self.loaded_inputs, axis=0)
            self.loaded_labels = np.concatenate(self.loaded_labels, axis=0)
            self.prev_load_pos = self.load_pos
            self.load_pos += increment
            self.pos = 0
        inputs, labels = self.loaded_inputs[self.pos:self.pos + self.batch_size], self.loaded_labels[self.pos:self.pos + self.batch_size]
        progress = self.prev_load_pos  # num files we've finished going through, not counting the current ones
        self.pos += self.batch_size
        return progress, inputs, labels
