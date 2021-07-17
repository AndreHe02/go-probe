import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from io_utils import *

def get_annotated_dataset(annotations, feature_func, label_func):
    if isinstance(annotations, str):
        annotations = read_pkl(annotations)
    return [(feature_func(ant), label_func(ant)) for ant in annotations]

import os
def save_dataset(dataset, path, bucket_size=1000):
    ft_file = "features_%d.npy"
    lbl_file = "labels_%d.npy"
    for bkt in range((len(dataset)-1) // bucket_size + 1):
        features = np.stack(ft for ft, _ in dataset[bkt*bucket_size:(bkt+1)*bucket_size])
        labels = np.stack(lbl for _, lbl in dataset[bkt*bucket_size:(bkt+1)*bucket_size])
        np.save(os.path.join(path, ft_file % bkt), features)
        np.save(os.path.join(path, lbl_file % bkt), labels)

def load_dataset(path):
    #order might change. lazy
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

def get_bow_dataset(annotations, keywords):
    return get_annotated_dataset(annotations, seven_planes, make_bag_of_words(keywords))

def get_bow_dataset_for_elf(annotations, keywords):
    return get_annotated_dataset(annotations, AGZ_features, make_bag_of_words(keywords))
