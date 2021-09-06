import numpy as np
from tqdm import tqdm
import os
import re
import pickle as pkl


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

def make_bag_of_words(words):
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

from go_probe.datasets.pattern_features import *
def patterns(ant):
    svp = seven_planes(ant)
    label = np.array([has_eye(svp), has_wall(svp), has_cut(svp), has_ladder(svp)])
    return label

def write_dataset(ants, feature_funcs, feature_names, path, bucket_size=1024):
    fname_bases = [f + "_%d.npy" for f in feature_names]
    for bkt in tqdm(range((len(ants)-1) // bucket_size + 1)):
        for fname_base, feature_func in zip(fname_bases, feature_funcs):
            fts = np.stack(feature_func(ant) for ant in ants[bkt*bucket_size:(bkt+1)*bucket_size])
            np.save(os.path.join(path, fname_base % bkt), fts)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='annotations_filtered.pkl')
    parser.add_argument('--keywords', default='top_go_words.txt')
    parser.add_argument('--controls1', default='top_control_words.txt')
    parser.add_argument('--controls2', default='sim_control_words.txt')
    parser.add_argument('-d', '--dataset_dir', default='C:/Users/andre/Documents/data/rerun/')
    args = parser.parse_args()

    with open(args.file, 'rb') as f:
        annotations = pkl.load(f)

    def read_words(fname):
        with open(fname, 'r') as f:
            return [w.strip() for w in f.readlines()]
    keywords = read_words(args.keywords)
    top_controls = read_words(args.controls1)
    sim_controls = read_words(args.controls2)
    probe_words = keywords + top_controls + sim_controls    
    
    print('Generating dataset')
    feature_funcs = [seven_planes, AGZ_features, make_bag_of_words(probe_words), patterns]
    feature_names = ["svp", "elf", "bow", "patterns"]
    write_dataset(annotations,  feature_funcs, feature_names, args.dataset_dir, 8192)
