import numpy as np
from torch.utils.data import Dataset
from dlgo import goboard_fast as goboard
from dlgo.gotypes import Point, Player
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.gosgf.sgf_properties import interpret_go_point
from tqdm import tqdm

def load_to_memory(dataset):
    return [x for x in tqdm(dataset)]

class SPBoWDataset(Dataset):
    
    def __init__(self, annotations, keywords):
        self.ants = annotations
        self.keywords = keywords
    
    @staticmethod
    def _get_seven_planes(sgf_str):
        game = goboard.GameState.new_game(19)        
        move_seq = sgf_str.split(';')[1:-1]
        for move_name in move_seq:
            idx = move_name.index('[')
            col_s, row_s = move_name[idx+1], move_name[idx+2]
            col = ord(col_s) - 97
            row = 19 - ord(row_s) + 96
            if 'B' in move_name:
                game.board.place_stone(Player.black, Point(row+1, col+1))
            elif 'W' in move_name:
                game.board.place_stone(Player.white, Point(row+1, col+1))
        encoder = SevenPlaneEncoder((19, 19))
        return encoder.encode(game)
    
    def get_bag_of_words(self, comment):
        comment = comment.lower()
        label = np.zeros(len(self.keywords))
        for i, kw in enumerate(self.keywords):
            if kw in comment:
                label[i] = 1
        return label
    
    def __len__(self):
        return len(self.ants)
    
    def __getitem__(self, idx):
        ant = self.ants[idx]
        return self._get_seven_planes(ant['sgf']), self.get_bag_of_words(ant['comments'])