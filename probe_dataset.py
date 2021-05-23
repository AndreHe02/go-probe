import numpy as np
from torch.utils.data import Dataset
from betago.dataloader.goboard import GoBoard
from betago.processor import SevenPlaneProcessor

spp = SevenPlaneProcessor()
class SPBoWDataset(Dataset):
    
    def __init__(self, annotations, keywords):
        self.ants = annotations
        self.keywords = keywords
    
    @staticmethod
    def _get_seven_planes(sgf):
        move_seq = sgf.split(';')[1:-1]
        go_board = GoBoard(19)
        for move in move_seq:
            if move[:2] == 'AW' or move[:2] == 'AB':
                color = move[1].lower()
                pos = (ord(move[3]) - 97, ord(move[4]) - 97)
            elif move[0] == 'W' or move[0] == 'B':
                color = move[0].lower()
                pos = (ord(move[2]) - 97, ord(move[3]) - 97)
            else:
                continue
            go_board.apply_move(color, pos)
        last_move = move_seq[-1]
        if 'W' in last_move:
            color = 'w'
        elif 'B' in last_move:
            color = 'b'
        else:
            raise ValueError()
        seven_planes, _ = spp.feature_and_label(color, (0, 0), go_board)
        seven_planes = np.concatenate((seven_planes, np.ones((1, 19, 19))), axis=0)
        return seven_planes
    
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