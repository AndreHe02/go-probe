import numpy as np
from torch.utils.data import Dataset
from dlgo import goboard_fast as goboard
from dlgo.gotypes import Point
from dlgo.encoders.sevenplane import SevenPlaneEncoder
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
            if 'AW' in move_name \
                or 'AB' in move_name \
                or 'W' in move_name \
                or 'B' in move_name:
                    idx = move_name.index('[')
                    row, col = ord(move[idx+1]) - 97, ord(move[idx+2]) - 97
                    move = goboard.Move(Point(row, col))
                    game.apply_move(move)       
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