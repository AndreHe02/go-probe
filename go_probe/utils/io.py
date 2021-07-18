import pickle as pkl
import os
import torch

def read_pkl(filename, mode='rb'):
    with open(filename, mode) as f:
        return pkl.load(f)

def write_pkl(obj, filename, mode='wb'):
    with open(filename, mode) as f:
        pkl.dump(obj, f)

def save_ckpt(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)

def load_ckpt(model, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])