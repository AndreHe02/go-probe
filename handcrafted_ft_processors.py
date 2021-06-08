import numpy as np
from betago.processor import SevenPlaneProcessor

class EyeProcessor(SevenPlaneProcessor):

    def __init__(self, data_directory='data', num_planes=7, consolidate=True, use_generator=False):
        super(EyeProcessor, self).__init__(data_directory=data_directory,
                                                  num_planes=num_planes,
                                                  consolidate=consolidate,
                                                  use_generator=use_generator)
        self.label_name = 'eyes'
        self.label_shape = (19, 19)

    def feature_and_label(self, color, move, go_board, num_planes):
        move_array, label = super().feature_and_label(color, move, go_board, num_planes)
        own = np.sum(move_array[0:3], axis=0)
        label_array = np.zeros((19, 19))
        for r in range(19):
            for c in range(19):
                if (r == 0 or own[r-1][c]) \
                    and (r == 18 or own[r+1][c]) \
                    and (c == 0 or own[r][c-1]) \
                    and (c == 18 or own[r][c+1]) \
                    and not own[r][c]:
                    label_array[r][c] = 1
        return move_array, label_array

class OppEyeProcessor(SevenPlaneProcessor):

    def __init__(self, data_directory='data', num_planes=7, consolidate=True, use_generator=False):
        super(OppEyeProcessor, self).__init__(data_directory=data_directory,
                                                  num_planes=num_planes,
                                                  consolidate=consolidate,
                                                  use_generator=use_generator)
        self.label_name = 'opp_eyes'
        self.label_shape = (19, 19)

    def feature_and_label(self, color, move, go_board, num_planes):
        move_array, label = super().feature_and_label(color, move, go_board, num_planes)
        own = np.sum(move_array[3:6], axis=0)
        label_array = np.zeros((19, 19))
        for r in range(19):
            for c in range(19):
                if (r == 0 or own[r-1][c]) \
                    and (r == 18 or own[r+1][c]) \
                    and (c == 0 or own[r][c-1]) \
                    and (c == 18 or own[r][c+1]) \
                    and not own[r][c]:
                    label_array[r][c] = 1
        return move_array, label_array

class WallProcessor(SevenPlaneProcessor):

    def __init__(self, data_directory='data', num_planes=7, consolidate=True, use_generator=False):
        super(WallProcessor, self).__init__(data_directory=data_directory,
                                                  num_planes=num_planes,
                                                  consolidate=consolidate,
                                                  use_generator=use_generator)
        self.label_name = 'walls'
        self.label_shape = (19, 19)

    def feature_and_label(self, color, move, go_board, num_planes):
        move_array, label = super().feature_and_label(color, move, go_board, num_planes)
        own = np.sum(move_array[0:3], axis=0)
        label_array = np.zeros(own.shape)
        minlen = 4
        for r in range(19):
            cont = 0
            for c in range(19):
                if own[r][c]:
                    cont += 1
                else:
                    cont = 0
                if cont == minlen:
                    for i in range(minlen):
                        label_array[r][c-i] = 1
                elif cont > minlen:
                    label_array[r][c] = 1
        for c in range(19):
            cont = 0
            for r in range(19):
                if own[r][c]:
                    cont += 1
                else:
                    cont = 0
                if cont == minlen:
                    for i in range(minlen):
                        label_array[r-i][c] = 1
                elif cont > minlen:
                    label_array[r][c] = 1
        return move_array, label_array

class OppWallProcessor(SevenPlaneProcessor):

    def __init__(self, data_directory='data', num_planes=7, consolidate=True, use_generator=False):
        super(OppWallProcessor, self).__init__(data_directory=data_directory,
                                                  num_planes=num_planes,
                                                  consolidate=consolidate,
                                                  use_generator=use_generator)
        self.label_name = 'opp_walls'
        self.label_shape = (19, 19)

    def feature_and_label(self, color, move, go_board, num_planes):
        move_array, label = super().feature_and_label(color, move, go_board, num_planes)
        own = np.sum(move_array[3:6], axis=0)
        label_array = np.zeros(own.shape)
        minlen = 4
        for r in range(19):
            cont = 0
            for c in range(19):
                if own[r][c]:
                    cont += 1
                else:
                    cont = 0
                if cont == minlen:
                    for i in range(minlen):
                        label_array[r][c-i] = 1
                elif cont > minlen:
                    label_array[r][c] = 1
        for c in range(19):
            cont = 0
            for r in range(19):
                if own[r][c]:
                    cont += 1
                else:
                    cont = 0
                if cont == minlen:
                    for i in range(minlen):
                        label_array[r-i][c] = 1
                elif cont > minlen:
                    label_array[r][c] = 1
        return move_array, label_array

class LastMoveProcessor(SevenPlaneProcessor):

    def __init__(self, data_directory='data', num_planes=7, consolidate=True, use_generator=False):
        super(LastMoveProcessor, self).__init__(data_directory=data_directory,
                                                  num_planes=num_planes,
                                                  consolidate=consolidate,
                                                  use_generator=use_generator)
        self.label_name = 'last_move'
        self.label_shape = ()

    def process_zip_full(self, dir_name, zip_file_name, data_file_name):
        # Read zipped file and extract name list
        import gzip
        import shutil
        import tarfile
        this_gz = gzip.open(dir_name + '/' + zip_file_name)
        this_tar_file = zip_file_name[0:-3]
        this_tar = open(dir_name + '/' + this_tar_file, 'wb')
        shutil.copyfileobj(this_gz, this_tar)  # random access needed to tar
        this_tar.close()
        this_zip = tarfile.open(dir_name + '/' + this_tar_file)
        name_list = this_zip.getnames()[1:]
        name_list.sort()

        # Determine number of examples
        total_examples = self.num_total_examples_full(this_zip, name_list)

        chunksize = 1024
        features = np.zeros((chunksize, self.num_planes, 19, 19))
        labels = np.zeros((chunksize, *self.label_shape))

        counter = 0
        chunk = 0

        feature_file_base = dir_name + '/' + data_file_name + '_features_%d'
        label_file_base = dir_name + '/' + data_file_name + '_' + self.label_name + '_%d'

        for name in name_list:
            if name.endswith('.sgf'):
                '''
                Load Go board and determine handicap of game, then iterate through all moves,
                store preprocessed move in data_file and apply move to board.
                '''
                sgf_content = this_zip.extractfile(name).read()
                sgf, go_board_no_handy = self.init_go_board(sgf_content)
                go_board, first_move_done = self.get_handicap(go_board_no_handy, sgf)
                first_move_done = False

                prev_move = None

                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()

                    if color is not None and move is not None:
                        row, col = move
                        if first_move_done:
                            X, y = self.feature_and_label(color, prev_move, go_board, self.num_planes)
                            features[counter%chunksize] = X
                            labels[counter%chunksize] = y
                            counter += 1

                            if counter % chunksize == 0:
                                feature_file = feature_file_base % chunk
                                label_file = label_file_base % chunk
                                chunk += 1
                                np.save(feature_file, features)
                                np.save(label_file, labels)

                        go_board.apply_move(color, (row, col))
                        first_move_done = True
                        prev_move = move
            else:
                raise ValueError(name + ' is not a valid sgf')

def is_surrounded(own, opp, i, j):
    marked = np.zeros((19, 19))
    if _is_surrounded(own, opp, i, j, marked):
        return marked
    else:
        return None

def _is_surrounded(own, opp, i, j, marked):
    if marked[i][j]:
        return True
    if opp[i][j]:
        return False
    if own[i][j]:
        return True
    marked[i][j] = 1
    adj_list = ((i+1, j), (i-1, j), (i, j-1), (i, j+1))
    for adj in adj_list:
        if adj[0] < 0 or adj[0] > 18 or adj[1] < 0 or adj[1] > 18:
            continue
        if not _is_surrounded(own, opp, adj[0], adj[1], marked):
            return False
    return True

def mark_surrounded(own, opp):
    marked = np.zeros((19, 19))
    for i in range(19):
        for j in range(19):
            if not marked[i][j]:
                group = is_surrounded(own, opp, i, j)
                if group is not None:
                    marked += group
    return marked

class SurroundProcessor(SevenPlaneProcessor):

    def __init__(self, data_directory='data', num_planes=7, consolidate=True, use_generator=False):
        super(SurroundProcessor, self).__init__(data_directory=data_directory,
                                                  num_planes=num_planes,
                                                  consolidate=consolidate,
                                                  use_generator=use_generator)
        self.label_name = 'surround'
        self.label_shape = (19, 19)

    def feature_and_label(self, color, move, go_board, num_planes):
        move_array, label = super().feature_and_label(color, move, go_board, num_planes)
        own = np.sum(move_array[0:3], axis=0)
        opp = np.sum(move_array[3:6], axis=0)
        label_array = mark_surrounded(own, opp)
        return move_array, label_array

class OppSurroundProcessor(SevenPlaneProcessor):

    def __init__(self, data_directory='data', num_planes=7, consolidate=True, use_generator=False):
        super(OppSurroundProcessor, self).__init__(data_directory=data_directory,
                                                  num_planes=num_planes,
                                                  consolidate=consolidate,
                                                  use_generator=use_generator)
        self.label_name = 'opp_surround'
        self.label_shape = (19, 19)

    def feature_and_label(self, color, move, go_board, num_planes):
        move_array, label = super().feature_and_label(color, move, go_board, num_planes)
        own = np.sum(move_array[0:3], axis=0)
        opp = np.sum(move_array[3:6], axis=0)
        label_array = mark_surrounded(opp, own)
        return move_array, label_array

class RankProcessor(SevenPlaneProcessor):

    def __init__(self, data_directory='data', num_planes=7, consolidate=True, use_generator=False):
        super(RankProcessor, self).__init__(data_directory=data_directory,
                                                  num_planes=num_planes,
                                                  consolidate=consolidate,
                                                  use_generator=use_generator)
        self.label_name = 'rank'
        self.label_shape = ()

    def feature_and_label(self, color, move, go_board, num_planes):
        move_array, label = super().feature_and_label(color, move, go_board, num_planes)
        #white is higher rank
        label = int(color == 'w')
        return move_array, label

class ResultProcessor(SevenPlaneProcessor):

    def __init__(self, data_directory='data', num_planes=7, consolidate=True, use_generator=False):
        super(ResultProcessor, self).__init__(data_directory=data_directory,
                                                  num_planes=num_planes,
                                                  consolidate=consolidate,
                                                  use_generator=use_generator)
        self.label_name = 'result'
        self.label_shape = ()

    def process_zip_full(self, dir_name, zip_file_name, data_file_name, write_fts=True):
        import gzip
        import shutil
        import tarfile
        # Read zipped file and extract name list
        this_gz = gzip.open(dir_name + '/' + zip_file_name)
        this_tar_file = zip_file_name[0:-3]
        this_tar = open(dir_name + '/' + this_tar_file, 'wb')
        shutil.copyfileobj(this_gz, this_tar)  # random access needed to tar
        this_tar.close()
        this_zip = tarfile.open(dir_name + '/' + this_tar_file)
        name_list = this_zip.getnames()[1:]
        name_list.sort()

        # Determine number of examples
        total_examples = self.num_total_examples_full(this_zip, name_list)

        chunksize = 1024
        if write_fts:
            features = np.zeros((chunksize, self.num_planes, 19, 19))
        labels = np.zeros((chunksize, *self.label_shape))

        counter = 0
        chunk = 0

        feature_file_base = dir_name + '/' + data_file_name + '_features_%d'
        label_file_base = dir_name + '/' + data_file_name + '_' + self.label_name + '_%d'

        for name in name_list:
            if name.endswith('.sgf'):
                '''
                Load Go board and determine handicap of game, then iterate through all moves,
                store preprocessed move in data_file and apply move to board.
                '''
                sgf_content = this_zip.extractfile(name).read()
                if b'RE[' in sgf_content:
                    re_idx = sgf_content.index(b'RE[')
                    end = sgf_content.find(b']', re_idx)
                    if b'W' in sgf_content[re_idx:end]:
                        label = 0
                    elif b'B' in sgf_content[re_idx:end]:
                        label = 1
                else:
                    label = 0

                sgf, go_board_no_handy = self.init_go_board(sgf_content)
                go_board, first_move_done = self.get_handicap(go_board_no_handy, sgf)
                first_move_done = False

                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None and move is not None:
                        row, col = move
                        if first_move_done:
                            X, _ = self.feature_and_label(color, move, go_board, self.num_planes)
                            if write_fts:
                                features[counter%chunksize] = X
                            if color == 'b':
                                y = label
                            else:
                                y = 1 - label
                            labels[counter%chunksize] = y
                            counter += 1

                            if counter % chunksize == 0:
                                feature_file = feature_file_base % chunk
                                label_file = label_file_base % chunk
                                chunk += 1
                                if write_fts:
                                    np.save(feature_file, features)
                                np.save(label_file, labels)


                        go_board.apply_move(color, (row, col))
                        first_move_done = True
            else:
                raise ValueError(name + ' is not a valid sgf')

class CapturedProcessor(SevenPlaneProcessor):

    def __init__(self, data_directory='data', num_planes=7, consolidate=True, use_generator=False):
        super(CapturedProcessor, self).__init__(data_directory=data_directory,
                                                  num_planes=num_planes,
                                                  consolidate=consolidate,
                                                  use_generator=use_generator)
        self.label_name = 'captured'
        self.label_shape = (19, 19)

    def process_zip_full(self, dir_name, zip_file_name, data_file_name, write_fts=True):
        import gzip
        import shutil
        import tarfile
        # Read zipped file and extract name list
        this_gz = gzip.open(dir_name + '/' + zip_file_name)
        this_tar_file = zip_file_name[0:-3]
        this_tar = open(dir_name + '/' + this_tar_file, 'wb')
        shutil.copyfileobj(this_gz, this_tar)  # random access needed to tar
        this_tar.close()
        this_zip = tarfile.open(dir_name + '/' + this_tar_file)
        name_list = this_zip.getnames()[1:]
        name_list.sort()

        # Determine number of examples
        total_examples = self.num_total_examples_full(this_zip, name_list)

        chunksize = 1024
        if write_fts:
            features = np.zeros((chunksize, self.num_planes, 19, 19))
        labels = np.zeros((chunksize, *self.label_shape))

        counter = 0
        chunk = 0

        feature_file_base = dir_name + '/' + data_file_name + '_features_%d'
        label_file_base = dir_name + '/' + data_file_name + '_' + self.label_name + '_%d'

        for name in name_list:
            if name.endswith('.sgf'):
                '''
                Load Go board and determine handicap of game, then iterate through all moves,
                store preprocessed move in data_file and apply move to board.
                '''
                sgf_content = this_zip.extractfile(name).read()
                sgf, go_board_no_handy = self.init_go_board(sgf_content)
                go_board, first_move_done = self.get_handicap(go_board_no_handy, sgf)
                first_move_done = False

                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None and move is not None:
                        row, col = move
                        if first_move_done:
                            X, _ = self.feature_and_label(color, move, go_board, self.num_planes)
                            if write_fts:
                                features[counter%chunksize] = X

                            captured = go_board.captured
                            y = np.zeros((19, 19))
                            for r, c in captured:
                                y[r, c] = 1

                            labels[counter%chunksize] = y
                            counter += 1

                            if counter % chunksize == 0:
                                feature_file = feature_file_base % chunk
                                label_file = label_file_base % chunk
                                chunk += 1
                                if write_fts:
                                    np.save(feature_file, features)
                                np.save(label_file, labels)


                        go_board.apply_move(color, (row, col))
                        first_move_done = True
            else:
                raise ValueError(name + ' is not a valid sgf')

def cuts(own, opp):

    def same_color_region(board, opp, i, j, stone_set, adj_set, marked):
        if marked[i][j] == 1:
            return
        marked[i][j] = 1
        stone_set.add((i, j))
        adj_list = ((i+1, j), (i-1, j), (i, j-1), (i, j+1))
        for adj in adj_list:
            if adj[0] < 0 or adj[0] > 18 or adj[1] < 0 or adj[1] > 18:
                continue
            if not board[adj[0]][adj[1]] and not opp[adj[0]][adj[1]]:
                adj_set.add(adj)
            if board[adj[0]][adj[1]] == board[i][j]:
                same_color_region(board, opp, adj[0], adj[1], stone_set, adj_set, marked)

    marked = np.zeros((19, 19))
    X = []
    marked = np.zeros((19, 19))
    for i in range(19):
        for j in range(19):
            if marked[i][j]:
                continue
            if opp[i][j]:
                s, a = set(), set()
                same_color_region(opp, own, i, j, s, a, marked)
                X.append((s, a))
    C = set()
    for ai, (s, a) in enumerate(X):
        for a2i, (s2, a2) in enumerate(X):
            if ai == a2i:
                continue
            if len(s) + len(s2) < 3:
                continue
            inter = set.intersection(a, a2)
            if len(inter) == 1:
                for r, c in inter:
                    libs = 0
                    if r > 0 and not opp[r-1][c]:
                        libs += 1
                    if r < 18 and not opp[r+1][c]:
                        libs += 1
                    if c > 0 and not opp[r][c-1]:
                        libs += 1
                    if c < 18 and not opp[r][c+1]:
                        libs += 1
                    if libs >= 2:
                        C.add((r, c))
    ft = np.zeros((19, 19))
    for i, j in C:
        ft[i][j] = 1
    return ft

class CutProcessor(SevenPlaneProcessor):

    def __init__(self, data_directory='data', num_planes=7, consolidate=True, use_generator=False):
        super(CutProcessor, self).__init__(data_directory=data_directory,
                                                  num_planes=num_planes,
                                                  consolidate=consolidate,
                                                  use_generator=use_generator)
        self.label_name = 'cuts'
        self.label_shape = (19, 19)

    def feature_and_label(self, color, move, go_board, num_planes):
        move_array, label = super().feature_and_label(color, move, go_board, num_planes)
        own = np.sum(move_array[0:3], axis=0)
        opp = np.sum(move_array[3:6], axis=0)
        label_array = cuts(own, opp)

        return move_array, label_array

def in_grid(i, j):
    return 0<=i<19 and 0<=j<19

def ladders(svp):
    own = np.sum(svp[0:3], axis=0)
    opp = np.sum(svp[3:6], axis=0)
    ft = np.zeros((2, 19, 19))
    for i in range(1, 18):
        for j in range(1, 18):
            if own[i][j] or opp[i][j]:
                continue
            continues_ladder = False
            libs = 0
            owns = 0
            adjs = ((i-1, j), (i+1, j), (i, j-1), (i, j+1))
            for r, c in adjs:
                if not in_grid(r, c):
                    continue
                if not own[r][c] and not opp[r][c]:
                    libs += 1
                if own[r][c]:
                    owns += 1
                if svp[0][r][c]:
                    continues_ladder = True
            if continues_ladder and libs == 2 and owns==1:
                ft[0][i][j] = 1
                if ladder_breaks(svp, i, j):
                    ft[1][i][j] = 1
    return ft

def zigzag(own, opp, i, j, step1, step2):
    r, c = i, j
    s = 1
    step = step2
    otherstep = step1
    while in_grid(r, c):
        if opp[r][c]:
            return False
        if own[r][c]:
            return True
        if not in_grid(r+step[0], c+step[1]) or opp[r+step[0]][c+step[1]]:
            return False
        if in_grid(r+step[0], c+step[1]) and own[r+step[0]][c+step[1]]:
            return True
        #cutting ladder breaker
        if in_grid(r+step[0]-otherstep[0], c+step[1]-otherstep[1]) and own[r+step[0]-otherstep[0]][c+step[1]-otherstep[1]]:
            if not in_grid(r-2*otherstep[0], c-2*otherstep[1]) or not opp[r-2*otherstep[0]][c-2*otherstep[1]]:
                return True
        if s % 2 == 0:
            step = step1
            otherstep = step2
        else:
            step = step2
            otherstep = step1
        r += step[0]
        c += step[1]
        s += 1
    return False

def ladder_breaks(svp, i, j):
    own = np.sum(svp[0:3], axis=0)
    opp = np.sum(svp[3:6], axis=0)

    #in grid
    if in_grid(i-1, j) and own[i-1][j]:
        step1 = (1, 0)
    elif in_grid(i+1, j) and own[i+1][j]:
        step1 = (-1, 0)
    elif in_grid(i, j-1) and own[i][j-1]:
        step1 = (0, 1)
    elif in_grid(i, j+1) and own[i][j+1]:
        step1 = (0, -1)

    if not in_grid(i-1, j) or opp[i-1][j]:
        step2 = (1, 0)
    elif not in_grid(i+1, j) or opp[i+1][j]:
        step2 = (-1, 0)
    elif not in_grid(i, j-1) or opp[i][j-1]:
        step2 = (0, 1)
    elif not in_grid(i, j+1) or opp[i][j+1]:
        step2 = (0, -1)

    #special case where you can zigzag both ways
    if step1[0] == -step2[0] and step1[1] == -step2[1]:
        zz1 = zigzag(own, opp, i, j, step1, (step1[1], step1[0]))
        zz2 = zigzag(own, opp, i, j, step1, (-step1[1], -step1[0]))
        return zz1 or zz2
    else:
        return zigzag(own, opp, i, j, step1, step2)

class LadderProcessor(SevenPlaneProcessor):
    def __init__(self, data_directory='data', num_planes=7, consolidate=True, use_generator=False):
        super(LadderProcessor, self).__init__(data_directory=data_directory,
                                                  num_planes=num_planes,
                                                  consolidate=consolidate,
                                                  use_generator=use_generator)
        self.label_name = 'ladders'
        self.label_shape = (2, 19, 19)

    def feature_and_label(self, color, move, go_board, num_planes):
        move_array, label = super().feature_and_label(color, move, go_board, num_planes)
        label_array = ladders(move_array)
        return move_array, label_array
