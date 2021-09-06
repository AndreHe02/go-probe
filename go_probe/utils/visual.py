import matplotlib.pyplot as plt
import numpy as np

    
def draw_go_board(b_board, w_board, feat):
    # create a 8" x 8" board
    if b_board is None:
        b_board = np.zeros((19, 19))
    if w_board is None:
        w_board = np.zeros((19, 19))
    if feat is None:
        feat = np.zeros((19, 19))
    
    fig = plt.figure(figsize=[8,8])
    fig.patch.set_facecolor((1,1,.8))
    ax = fig.add_subplot(111)

    # draw the grid
    for x in range(19):
        ax.plot([x, x], [0,18], 'k')
    for y in range(19):
        ax.plot([0, 18], [y,y], 'k')

    # scale the axis area to fill the whole figure
    ax.set_position([0,0,1,1])

    # get rid of axes and everything (the figure background will show through)
    ax.set_axis_off()

    # scale the plot area conveniently (the board is in 0,0..18,18)
    ax.set_xlim(-1,19)
    ax.set_ylim(-1,19)

    def draw_stone(row, col, color):
        if color == 'b':
            ax.plot(row,col,'o',markersize=25, markeredgecolor=(0,0,0), markerfacecolor='k', markeredgewidth=2)
        elif color == 'w':
            ax.plot(row,col,'o',markersize=25, markeredgecolor=(.5,.5,.5), markerfacecolor='w', markeredgewidth=2)

    for i in range(len(b_board)):
        for j in range(len(b_board[0])):
            if b_board[i][j]:
                draw_stone(i, j, 'b')
            if w_board[i][j]:
                draw_stone(i, j, 'w')
                
    for r in range(19):
        for c in range(19):
            ax.plot(r, c, 'X', markersize=feat[r][c], markerfacecolor='r')
    return ax
