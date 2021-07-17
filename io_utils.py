import pickle as pkl

def read_pkl(filename, mode='rb'):
    with open(filename, mode) as f:
        return pkl.load(f)

def write_pkl(obj, filename, mode='wb'):
    with open(filename, mode) as f:
        pkl.dump(obj, f)

        