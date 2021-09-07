from dlgo import goboard_fast as goboard
from dlgo.gotypes import Point, Player
from tqdm import tqdm

def filter_annotations(ants):
    '''
    Samples with wrong board sizes, invalid move sequences, or empty boards are removed.
    We also skip all samples with the "Add Empty" action in their move sequences. 
    "AE" is used to remove a piece in order to show an alternative move sequence, 
    but our GameState objects don't support this.
    '''    
    def get_game_state(sgf):
        game = goboard.GameState.new_game(19)
        move_seq = sgf.split(';')[1:-1]
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
        return game

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
            game = get_game_state(ant['sgf'])
        except Exception as e:
            skipped['moves'] += 1
            continue
        if not game.board or not ('W' in ant['sgf'] or 'B' in ant['sgf']):
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
    import random
    random.seed(0)
    random.shuffle(filtered)
    with open(output, 'wb') as f:
        pkl.dump(filtered, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='annotations.pkl')
    parser.add_argument('-o', '--output', default='annotations_filtered.pkl')
    args = parser.parse_args()

    main(args.file, args.output)