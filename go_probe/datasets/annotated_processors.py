import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from go_probe.utils.io import *
import os
import random
import torch
import re

def get_annotated_dataset(annotations, feature_func, label_func):
    if isinstance(annotations, str):
        annotations = read_pkl(annotations)
    return [(feature_func(ant), label_func(ant)) for ant in annotations]

def write_dataset(ants, feature_func, label_func, path, bucket_size=1000):
    if isinstance(ants, str):
        ants = read_pkl(ants)
    ft_file = "features_%d.npy"
    lbl_file = "labels_%d.npy"
    for bkt in range((len(ants)-1) // bucket_size + 1):
        features = np.stack(feature_func(ant) for ant in ants[bkt*bucket_size:(bkt+1)*bucket_size])
        labels = np.stack(label_func(ant) for ant in ants[bkt*bucket_size:(bkt+1)*bucket_size])
        np.save(os.path.join(path, ft_file % bkt), features)
        np.save(os.path.join(path, lbl_file % bkt), labels)

def load_whole_dataset(path):
    #order might change. whatever
    features, labels = [], []
    for file in os.listdir(path):
        if not 'features' in file:
            continue
        features.append(np.load(os.path.join(path, file)))
        labels.append(np.load(os.path.join(path, file.replace('features', 'labels'))))
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return [(features[i], labels[i]) for i in range(len(features))]

from dlgo import goboard_fast as goboard
from dlgo.gotypes import Point, Player
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.gosgf.sgf_properties import interpret_go_point

def seven_planes(ant):
    sgf_str = ant['sgf']
    game = goboard.GameState.new_game(19)
    move_seq = sgf_str.split(';')[1:-1]
    for move_name in move_seq:
        idx = move_name.index('[')
        col_s, row_s = move_name[idx+1], move_name[idx+2]
        col = ord(col_s) - 97
        row = 19 - ord(row_s) + 96
        if 'B' in move_name:
            game.board.place_stone(Player.black, Point(row+1, col+1))
            game.next_player = Player.white
        elif 'W' in move_name:
            game.board.place_stone(Player.white, Point(row+1, col+1))
            game.next_player = Player.black
    encoder = SevenPlaneEncoder((19, 19))
    svp = encoder.encode(game)
    ones = np.ones((1, 19, 19))
    return np.concatenate((svp, ones), axis=0)

def AGZ_features(ant):
    sgf_str = ant['sgf']
    game = goboard.GameState.new_game(19)
    move_seq = sgf_str.split(';')[1:-1]
    AGZ_feat = np.zeros((18, 19, 19), dtype=np.float64)
    encoder = SevenPlaneEncoder((19, 19))

    for mv, move_name in enumerate(move_seq):
        idx = move_name.index('[')
        col_s, row_s = move_name[idx+1], move_name[idx+2]
        col = ord(col_s) - 97
        row = 19 - ord(row_s) + 96
        if 'B' in move_name:
            game.board.place_stone(Player.black, Point(row+1, col+1))
            game.next_player = Player.white
        elif 'W' in move_name:
            game.board.place_stone(Player.white, Point(row+1, col+1))
            game.next_player = Player.black

        t = len(move_seq)-mv-1
        if t < 8:
            svp = encoder.encode(game)
            own = np.sum(svp[0:3], axis=0)
            opp = np.sum(svp[3:6], axis=0)
            if t % 2 == 1:
                own, opp = opp, own
            AGZ_feat[t*2] = own
            AGZ_feat[t*2+1] = opp

        if game.next_player == Player.black:
            AGZ_feat[16] = 1
        else:
            AGZ_feat[17] = 1

    return AGZ_feat

def make_bag_of_words(keywords):
    def bag_of_words(ant):
        nonlocal keywords
        comment = ant['comments']
        comment = comment.lower()
        label = np.zeros(len(keywords))
        for i, kw in enumerate(keywords):
            if kw in comment:
                label[i] = 1
        return label
    return bag_of_words

def better_bag_of_words(words):
    def bow(ant):
        nonlocal words
        cmt = ant['comments']
        cmt = cmt.lower()
        label = np.zeros(len(words))
        for word in re.sub('[^A-Za-z0-9 ]+', '', cmt.lower()).split(' '):
            if word in words:
                label[words.index(word)] = 1
        return label
    return bow

def get_bow_dataset(annotations, keywords):
    return get_annotated_dataset(annotations, seven_planes, make_bag_of_words(keywords))

def get_bow_dataset_for_elf(annotations, keywords):
    return get_annotated_dataset(annotations, AGZ_features, make_bag_of_words(keywords))

# use dataset.CrossValDataset instead
def disk_data_loader(path, batch_size, shuffle=False):
    fnames = [f for f in os.listdir(path) if 'features' in f]
    if shuffle:
        random.shuffle(fnames)

    def batch_generator(batch_size):
        for fname in fnames:
            features = np.load(os.path.join(path, fname))
            labels = np.load(os.path.join(path, fname.replace('features', 'labels')))
            for bkt in range(len(features) // batch_size):
                X, y = features[bkt*batch_size:(bkt+1)*batch_size], \
                    labels[bkt*batch_size:(bkt+1)*batch_size]
                X = torch.from_numpy(X).type(torch.FloatTensor)
                y = torch.from_numpy(y).type(torch.FloatTensor)
                yield X, y

    return batch_generator(batch_size)

if __name__ == "__main__":
    annotations = read_pkl('C:/users/andre/documents/data/go/annotated/annotations_filtered.pkl')
    go_dict = read_pkl('C:/users/andre/documents/data/go/annotated/sorted_go_dict.pkl')
    keywords = go_dict[:30]
    write_dataset(annotations, seven_planes, better_bag_of_words(keywords), 'C:/users/andre/documents/data/go/annotated/go_bow/')
    write_dataset(annotations, AGZ_features, better_bag_of_words(keywords), 'C:/users/andre/documents/data/go/annotated/elf_bow/')