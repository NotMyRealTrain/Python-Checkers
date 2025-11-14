import numpy as np
from checkers.constants import ROWS, COLS, RED, WHITE

def board_to_state(board):
    # Returns a 1D numpy array of shape (ROWS * COLS * 5,)
    # where each square is either empty, red_piece, red_king, white_piece, white_king
    state = np.zeros((ROWS, COLS, 5), dtype=np.float32)
    for row in range(ROWS):
        for col in range(COLS):
            piece = board.get_piece(row, col)
            if piece == 0:
                state[row, col, 0] = 1  # empty
            else:
                if piece.color == RED:
                    if piece.king:
                        state[row, col, 2] = 1  # red king
                    else:
                        state[row, col, 1] = 1  # red piece
                elif piece.color == WHITE:
                    if piece.king:
                        state[row, col, 4] = 1  # white king
                    else:
                        state[row, col, 3] = 1  # white piece
    return state.flatten()

def get_valid_actions(board, color):
    # Get list of valid actions for a color
    # Each action is a tuple (piece_row, piece_col, move_row, move_col)
    actions = []
    for piece in board.get_all_pieces(color):
        valid_moves = board.get_valid_moves(piece)
        for move, skip in valid_moves.items():
            actions.append((piece.row, piece.col, move[0], move[1]))
    return actions

def apply_action(board, action, color):
    # apply an action to the board and return the new board
    # Action is (piece_row, piece_col, move_row, move_col)
    from copy import deepcopy
    new_board = deepcopy(board)
    piece_row, piece_col, move_row, move_col = action
    piece = new_board.get_piece(piece_row, piece_col)
    valid_moves = new_board.get_valid_moves(piece)
    skipped = valid_moves.get((move_row, move_col), [])
    new_board.move(piece, move_row, move_col)
    if skipped:
        new_board.remove(skipped)
    return new_board