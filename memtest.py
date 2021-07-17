from annotated_datasets import *
keywords = ['territory', 'cut', 'sente', 'shape', 'moyo',
            'ko', 'invasion', 'influence', 'wall', 'eye']
sgf_dataset = get_annotated_dataset('data/contains_top_10_kw.pkl', lambda x: x, make_bag_of_words(keywords))

from elf.df_model3 import Model_PolicyValue

class ModelOptions:
    leaky_relu = False
    dim = 256
    bn = True
    bn_momentum = 0.1
    bn_eps = 1e-5
    num_block = 20
    gpu = 0
    use_data_parallel = False
    use_data_parallel_distributed = False

params = {"board_size": 19, "num_planes":18}

model = Model_PolicyValue(ModelOptions, params)

_replace_prefix = ["resnet.module,resnet", "init_conv.module,init_conv"]
replace_prefix = [
    item.split(",")
    for item in _replace_prefix
]
model.load(
    "elf/pretrained-go-19x19-v2.bin",
    omit_keys=[],
    replace_prefix=replace_prefix,
    check_loaded_options=False)
model = model.cuda()

def AGZ_features(sgf_str):
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

def agz_reprs(X):
    agz_feats = [AGZ_features(X['sgf'][i]) for i in range(len(X['sgf']))]
    agz_feats = np.stack(agz_feats)
    rep = model.forward_resnet(torch.from_numpy(agz_feats).type(torch.FloatTensor).cuda())
    return [rep.detach()]

import torch.nn as nn
import torch.optim as optim
init_probes = lambda : [nn.Sequential(nn.Flatten(), nn.Linear(256*19*19, len(keywords))).cuda()]
criterion = nn.BCEWithLogitsLoss()
num_epochs = 5

from experiments import *
def cross_validation_run(dataset, get_reprs, init_probes, name, log_dir):
    metric_dfs = []
    for val_idx in range(10):
        chunk_size = len(dataset)//10
        train_dataset = dataset[:val_idx*chunk_size] + dataset[(val_idx+1)*chunk_size:]
        test_dataset = dataset[val_idx*chunk_size:(val_idx+1)*chunk_size]

        probes = init_probes()
        optimizers = [optim.Adam(probes[i].parameters(), lr=0.001) for i in range(len(probes))]

        probe_layers_parallel(train_dataset, test_dataset, get_reprs, probes, optimizers, criterion, num_epochs, log_dir%val_idx)
        df = evaluate_auc(test_dataset, get_reprs, probes)
        df['val'] = val_idx
        metric_dfs.append(df)
    metric_df = pd.concat(metric_dfs)
    metric_df['name'] = name
    return metric_df

aucs = cross_validation_run(sgf_dataset, agz_reprs, init_probes, 'elf_resnet', 'temp/cv%d')
