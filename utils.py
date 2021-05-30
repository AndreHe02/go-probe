import pickle as pkl

def read_pkl(filename, mode='rb'):
    with open(filename, mode) as f:
        return pkl.load(f)

def write_pkl(filename, obj, mode='wb'):
    with open(filename, mode) as f:
        pkl.dump(f, obj)

        