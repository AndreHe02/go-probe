import numpy as np

def has_wall(svp):
    own = np.sum(svp[0:3], axis=0)
    minlen = 4
    for r in range(19):
        cont = 0
        for c in range(19):
            if own[r][c]:
                cont += 1
            else:
                cont = 0
            if cont >= minlen:
                return True
    for c in range(19):
        cont = 0
        for r in range(19):
            if own[r][c]:
                cont += 1
            else:
                cont = 0
            if cont >= minlen:
                return True
    return False

def has_eye(svp):

    def is_surrounded(own, opp, i, j, visited):
        if visited[i][j]:
            return True
        if opp[i][j]:
            return False
        if own[i][j]:
            return True
        visited[i][j] = 1
        adj_list = ((i+1, j), (i-1, j), (i, j-1), (i, j+1))
        for adj in adj_list:
            if adj[0] < 0 or adj[0] > 18 or adj[1] < 0 or adj[1] > 18:
                continue
            if not is_surrounded(own, opp, adj[0], adj[1], visited):
                return False
        return True

    own = np.sum(svp[0:3], axis=0)
    opp = np.sum(svp[3:6], axis=0)
    
    marked = np.zeros((19, 19))
    for i in range(19):
        for j in range(19):
            if marked[i][j]:
                continue 
            visited = np.zeros((19, 19))
            if is_surrounded(own, opp, i, j, visited):
                return True
            else:
                marked += visited
    return False

def has_cut(svp):

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

    own = np.sum(svp[0:3], axis=0)
    opp = np.sum(svp[3:6], axis=0)

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
                        return True
    return False

def has_ladder(svp):

    def in_grid(i, j):
        return 0<=i<19 and 0<=j<19

    own = np.sum(svp[0:3], axis=0)
    opp = np.sum(svp[3:6], axis=0)

    ft = np.zeros((19, 19))
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
                return True
    return False
