from collections import defaultdict
from betago.dataloader.goboard import GoBoard
from tqdm import tqdm

def filter_annotations(ants):
    '''
    Samples with wrong board sizes, invalid move sequences, or empty boards are removed.
    We also skip all samples with the "Add Empty" action in their move sequences. 
    "AE" is used very rarely by some samples to remove a piece in order to show an alternative move sequence (backtracking), 
    but this is annoying to implement on GoBoard objects.
    '''    
    def get_board_state(sgf):
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
        return go_board
    filtered = []
    skipped = {'size':0, 'moves':0, 'backtrack':0, 'empty':0}
    for ant in tqdm(ants):
        if 'AE' in ant['sgf']:
            skipped['backtrack'] += 1
            continue
        if len(ant['board_state'][0]) != 19:
            skipped['size'] += 1
            continue
        try:
            go_board = get_board_state(ant['sgf'])
        except Exception as e:
            skipped['moves'] += 1
            continue
        if not go_board.board or not ('W' in ant['sgf'] or 'B' in ant['sgf']):
            skipped['empty'] += 1
            continue
        filtered.append(ant)
    print('Skipped for reasons:', skipped)
    print('%d of %d samples usable' % (len(filtered), len(ants)))
    return filtered

def main(fname, output):
    import pickle as pkl
    ants = pkl.load(open(fname, 'rb'))
    filtered = filter_annotations(ants)
    with open(output, 'wb') as f:
        pkl.dump(filtered, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='annotations.pkl')
    parser.add_argument('-o', '--output', default='annotations_filtered.pkl')
    args = parser.parse_args()

    main(args.file, args.output)