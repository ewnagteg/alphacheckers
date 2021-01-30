import numpy as np
# Move:
# canmove  jumped   origin   to
#    1      11111   11111 11111
#
# canmove: 0x8000
#   -> if this move is played, player can still jump after it nd it is therefore still their turn
# to: 0x1f
#   -> Where piece jumps to
# origin: 0x3e0 >> 5
#   -> Where piece jumped from
# jumped: 0x7C00  >> 10
#   -> 0 if no piece jumped (since 0 square can't be jumped)
#
# For move_index (index of move in pi vector)
# only uses "origin" and "to"
# ie just move&0x3FF




# State:
# [p1, p1k, p2, p2k, player_turn]
# p1: bitboard of player 1's pieces
# p1k: bitboard of player 1's kings
# p2: bitboard of player 2's pieces
# p2k: bitboard of player 2's kings
# player_turn: 1 or -1 => 1 is player 1, -1 is player 2

# Board
#    0  1  2  3  4  5  6  7
# 0|    4    12    20    28
# 1| 0     8    16    24
# 2|    5    13    21    29
# 3| 1     9    17    25
# 4|    6    14    22    30
# 5| 2    10    18    26
# 6|    7    15    23    31
# 7| 3    11    19    27S

# Starting positions:
# Player 1 (1, Top)
#  4, 12, 20, 28, 0, 8, 16, 24, 5, 13, 21, 29
# Player 2 (-1, Bottom)
#  3, 11, 19, 27, 7, 15, 23, 31, 2, 10, 18, 26
# 
# Player 1 starts game.

def isOdd(n):
    return n % 2 == 1

def encode(x: int, y: int):
    return (8*x + y) >> 1

def getX(index):
    return index >> 2

def getY(index):
    return ((index & 3) << 1) + 1 - ((index >> 2) & 1)

def is_square(x, y):
    return ((x % 2) == 1) != ((y % 2) == 1)

def can_capture(peice, p1p, p2p, captures, player, king, recourse=True):
    board = p1p|p2p
    x = getX(peice)
    y = getY(peice)
    if x > 1: # jump left
        if (player == 1 or king) and y < 6: # down
            jumped = ((8*(x-1) + (y+1)) >> 1)
            jn = 1 << jumped
            target = (8*(x-2) + (y+2)) >> 1
            tn = 1 << target
            if 0 == (board & tn) and ((jn & p2p) == jn if player==1 else (jn & p1p) == jn):
                if recourse:
                    vector = (peice << 5) | target | (jumped << 10) 
                    if can_capture(target, p1p&(~(1 << jumped)), p2p&(~(1 << jumped)), captures, player, king, recourse=False):
                        vector |= 0x8000 # Canmove
                    captures.append(vector)
                else:
                    return True
        if (player == -1 or king) and y > 1: # up
            jumped = ((8*(x-1) + (y-1)) >> 1)
            jn = 1 << jumped
            target = (8*(x-2) + (y-2)) >> 1
            tn = 1 << target
            if 0 == (board & tn) and ((jn & p2p) == jn if player==1 else (jn & p1p) == jn):
                if recourse:    
                    vector = (peice << 5) | target | (jumped << 10)  
                    if can_capture(target, p1p&(~(1 << jumped)), p2p&(~(1 << jumped)), captures, player, king, recourse=False):
                        vector |= 0x8000 # Canmove
                    captures.append(vector)
                else:
                    return True
    if x < 6: # right
        if (player == 1 or king) and y < 6: # down
            jumped = ((8*(x+1) + (y+1)) >> 1)
            jn = 1 << jumped
            target = (8*(x+2) + (y+2)) >> 1
            tn = 1 << target
            if 0 == (board & tn) and ((jn & p2p) == jn if player==1 else (jn & p1p) == jn):
                if recourse:
                    vector = (peice << 5) | target | (jumped << 10)  
                    if can_capture(target, p1p&(~(1 << jumped)), p2p&(~(1 << jumped)), captures, player, king, recourse=False):
                        vector |= 0x8000 # Canmove
                    captures.append(vector)
                else:
                    return True
        if (player != 1 or king) and y > 1: # up
            jumped = ((8*(x+1) + (y-1)) >> 1)
            jn = 1 << jumped
            target = (8*(x+2) + (y-2)) >> 1
            tn = 1 << target
            if 0 == (board & tn) and ((jn & p2p) == jn if player==1 else (jn & p1p) == jn):
                if recourse:
                    vector = (peice << 5) | target | (jumped << 10)  
                    if can_capture(target, p1p&(~(1 << jumped)), p2p&(~(1 << jumped)), captures, player, king, recourse=False):
                        vector |= 0x8000 # Canmove
                    captures.append(vector)
                else:
                    return True
    return False
                
def get_moves(p1: int, p1k: int, p2: int, p2k: int, player: int):
    """
    Parameters:
    p1 (int): bitboard of player 1's pieces
    p1k (int): bitboard of player 1's kings
    p2 (int): bitboard of player 2's pieces
    p2k (int): bitboard of player 2's kings
    player (int): 1 or -1 => 1 is player 1, -1 is player 2

    Returns: list of moves
    """
    moves = []
    captures = []
    c = p1|p1k if player > 0 else p2|p2k
    p1p = p1|p1k
    p2p = p2|p2k
    board = p1p|p2p
    while c != 0:
        # binary search for a peice
        i = 32
        v = c
        v &= -v
        if (v): i-=1
        if (v & 0x0000FFFF): i -= 16
        if (v & 0x00FF00FF): i -= 8
        if (v & 0x0F0F0F0F): i -= 4
        if (v & 0x33333333): i -= 2
        if (v & 0x55555555): i -= 1
        # blank out peice
        c &= (~(1 << i))
        x = i >> 2 # getX
        y =  ((i & 3) << 1) + 1 - ((i >> 2) & 1) #getY
        n = 1 << i
        # if (p1 & p1k) == 0, no peices
        king = ((p1k|p2k)&n)==n

        # check captures
        can_capture(i, p1p, p2p, captures, player, king)

        if len(captures) == 0:
            if x > 0: # left
                if (player == 1 or king) and y < 7: # down
                    target = (8*(x-1) + (y+1)) >> 1
                    tn = 1 << target
                    if (board & tn) == 0:
                        moves.append((i<<5)|target)
                if (player != 1 or king) and y > 0: # up
                    target = (8*(x-1) + (y-1)) >> 1
                    tn = 1 << target
                    if (board & tn) == 0:
                        moves.append((i<<5)|target)
            if x < 7:
                if (player == 1 or king) and y < 7: # down
                    target = (8*(x+1) + (y+1)) >> 1
                    tn = 1 << target
                    if (board & tn) == 0:
                        moves.append((i<<5)|target)
                if (player != 1 or king) and y > 0: # up
                    target = (8*(x+1) + (y-1)) >> 1
                    tn = 1 << target
                    if (board & tn) == 0:
                        moves.append((i<<5)|target)
    return moves if len(captures) == 0 else captures

def run_move(p1, p1k, p2, p2k, player, move):
    to = move & 0x1f
    y = getY(to)
    from_ = (move&0x3e0) >> 5
    cap = (move&0x7c00) >> 10
    canmove = (move&0x8000) >> 15
    fn = 1 << from_
    king = ((p1k|p2k)&fn==fn)
    if player == 1:
        if cap != 0:
            p2 &= ~(1<<cap)
            p2k &= ~(1<<cap)
        p1 &= ~(1<<from_)
        p1k &= ~(1<<from_)
        if king or y==7:
            p1k |= 1<<to
        else:
            p1 |= 1<<to
    else:
        if cap != 0:
            p1 &= ~(1<<cap)
            p1k &= ~(1<<cap)
        p2 &= ~(1<<from_)
        p2k &= ~(1<<from_)
        if king or y==0:
            p2k |= 1<<to
        else:
            p2 |= 1<<to
    if canmove == 0:
        player *= -1
    return [p1, p1k, p2, p2k, player]


def gen_start():
    p1_ = 0
    p1 = [
        4, 12, 20, 28,
        0, 8, 16, 24, 
        5, 13, 21, 29
    ]
    
    for i in p1:
        p1_ |= 1 << i

    p2 = [
        3, 11, 19, 27,
        7, 15, 23, 31,
        2, 10, 18, 26
    ]

    p2_ = 0
    for i in p2:
        p2_ |= 1 << i
    return [p1_, 0, p2_, 0, 1]

def move_to_index(move: int, player):
    """
    Parameters:
    move: move to get index for
    player: current player's turn, important since for player = -1, the move has to be reflected
            ie move = 31 - move
    
    Returns:
        Index for move, where the index is the index of the policy vector.
    """
    if player == -1:
        move = reflect_move(move)
    move = move & 0x3FF
    to = move & 0x1f
    orgin = (move&0x3e0) >> 5
    index = orgin * 8

    x_0 = getX(orgin)
    x = getX(to)
    y_0 = getY(orgin)
    y = getY(to)

    vector = [x-x_0, y-y_0]
    cap = abs(vector[0]) > 1 or abs(vector[1]) > 1
    # norm vector to len 1
    if cap:
        vector[0] /= 2
        vector[1] /= 2

    # translate to q1
    vector[0] += 1
    vector[1] += 1

    # norm vector
    vector[0] /= 2
    vector[1] /= 2

    # calculate offset by mapping vector to 1d
    offset = vector[0] + 2*vector[1]
    # map jumps to diff offset
    offset += 4 if cap else 0

    index += offset
    return int(index)


def get_moves_mask(state, player):
    mask = np.ones((32*8, 1), dtype=bool)
    valid = get_moves(*state)
    # list of ints corresponding to move index's of moves (for policy head, see index_to_move ect for spec)
    moves = [move_to_index(i, player) for i in valid]
    mask[moves] = False
    return mask, moves

def index_to_move(index: int, player: int):
    orgin = int(index // 8)
    x = getX(orgin)
    y = getY(orgin)
    vector = index % 8
    cap = vector > 3
    if cap:
        vector %= 4
    vx = int(vector % 2)
    vy = int(vector // 2)
    # scale
    vx *= 2
    vy *= 2
    # translate
    vx -= 1
    vy -= 1
    if cap:
        vx *= 2
        vy *= 2
    to = encode(x+vx, y+vy)
    move = (to|(orgin<<5))
    # we ignore captured peice and whether we can jump again
    # ie dont use this method to get a move from a index, 
    # since it's missing 6 bits of info necessary (canjump + jumped peice)
    # you can compare by doing: "(move & 0x3FF) == index_to_move(index)"
    if player == -1:
        move = reflect_move(move)
    return move

def hard_index_to_move(index: int, moves, player):
    """
    Necessary since the index ignores whether a piece can still jump after the move.
    """
    move = index_to_move(index, player)
    for m in moves:
        if (m&0x3FF) == move:
            return m
    
def render(p1, p1k, p2, p2k, player):
    state = {}

    squares = []
    for i in range(32):
        squares.append(i)

    for sq in squares:
        n = 1 << sq
        if (n & p1) == n:
            state[sq] = 1
        elif (n & p1k) == n:
            state[sq] = 2
        elif (n & p2) == n:
            state[sq] = -1
        elif (n & p2k) == n:
            state[sq] = -2
     
    board = ''
    for y in range(8):
        line = ''
        for x in range(8):
            symb = '   '
            if (is_square(x, y)):
                i = encode(x, y)
                symb = ' . '
                if (i in state):
                    if (state[i] > 0):
                        symb = '@'
                    else:
                        symb = '%'
                    if (abs(state[i]) > 1):
                        symb = '[' + symb + ']'
                    else:
                        symb = ' ' + symb + ' '
            line += symb
        board += line + '\n'
    print(board)
    print("Player turn: {}".format(player))
    print()

def pretty_print_move(move):
    to = (move & 0x1f)
    can_move = move & 0x8000
    jumped = (move & 0x7C00)  >> 10
    orgin = ((move & 0x3e0) >> 5)
    print("from: {}\nto: {}\njumped: {}\ncanmove: {}\n\n".format(orgin, to, jumped, can_move))
    return move

def winner(p1, p1k, p2, p2k, player):
    """
    Parameters:
    p1 (int): bitboard of player 1's pieces
    p1k (int): bitboard of player 1's kings
    p2 (int): bitboard of player 2's pieces
    p2k (int): bitboard of player 2's kings
    player (int): 1 or -1 => 1 is player 1, -1 is player 2

    Should only be called when game is over, ie if max turns have been exceeded or one player cannot move.
    This is a stopgap solution for forcing games to end after about 80 turns, since this makes training easier.
    """
    if len(get_moves(p1, p1k, p2, p2k, player)) == 0:
        return player * -1
    return 0


def count_peices(n):
    count = 0
    while n != 0:
        n = n & (n-1)
        count += 1
    return count

def get_state_key(p1, p1k, p2, p2k, player):
    return "{} {} {} {} {}".format(p1, p1k, p2, p2k, player)

def reflect(n):
    """
    Parameters:
    n: square to reflect
    Returns:
    Where the square is from perspective of opposite player
    Equivalent to doing encode(7-x, 7-y) where (x, y) are coordinates of "n"
    """
    return 31-n

def reflect_move(move):
    to = 31 - (move & 0x1f)
    can_move = move & 0x8000
    jumped = (move & 0x7C00)  >> 10
    jumped = 31- jumped if jumped != 0 else 0
    orgin = 31 - ((move & 0x3e0) >> 5)
    return can_move | (jumped << 10) | (orgin << 5) | to

