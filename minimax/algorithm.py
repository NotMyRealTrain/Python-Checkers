## https://www.youtube.com/watch?v=l-hh51ncgDI minimax with alpha-beta pruning
from copy import deepcopy
import pygame

RED = (255,0,0)
WHITE = (255, 255, 255)

def minimax(position, depth, max_player, game, alpha=float('-inf'), beta=float('inf')):
    if depth == 0 or position.winner() != None:
        return position.evaluate(), position
    
    if max_player:
        maxEval = float('-inf')
        best_move = None
        for move in get_all_moves(position, WHITE, game):
            evaluation = minimax(move, depth-1, False, game, alpha, beta)[0]
            maxEval = max(maxEval, evaluation)
            if maxEval == evaluation:
                best_move = move
            
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break  # Beta cutoff (alpha-beta pruning)
        
        return maxEval, best_move
    else:
        minEval = float('inf')
        best_move = None
        for move in get_all_moves(position, RED, game):
            evaluation = minimax(move, depth-1, True, game, alpha, beta)[0]
            minEval = min(minEval, evaluation)
            if minEval == evaluation:
                best_move = move
            
            beta = min(beta, evaluation)
            if beta <= alpha:
                break  # Alpha cutoff (alpha-beta pruning)
        
        return minEval, best_move


def simulate_move(piece, move, board, game, skip):
    board.move(piece, move[0], move[1])
    if skip:
        board.remove(skip)

    return board


def get_all_moves(board, color, game):
    moves = []

    for piece in board.get_all_pieces(color):
        valid_moves = board.get_valid_moves(piece)
        for move, skip in valid_moves.items():
            draw_moves(game, board, piece)
            temp_board = deepcopy(board)
            temp_piece = temp_board.get_piece(piece.row, piece.col)
            new_board = simulate_move(temp_piece, move, temp_board, game, skip)
            moves.append(new_board)

        # Sort moves by evaluation (better moves first for alpha-beta pruning)
    if color == WHITE:
        moves.sort(key=lambda x: x.evaluate(), reverse=True)  # Best moves first for maximizer [10, 2, -5]
    else:
        moves.sort(key=lambda x: x.evaluate())  # Best moves first for minimizer [-5, 2, 10]
    
    return moves


def draw_moves(game, board, piece):
    valid_moves = board.get_valid_moves(piece)
    board.draw(game.win)
    pygame.draw.circle(game.win, (0,255,0), (piece.x, piece.y), 50, 5)
    game.draw_valid_moves(valid_moves.keys())
    pygame.display.update()
    #pygame.time.delay(100)

def minimax_alpha_beta(position, depth, max_player, game):

    # Convenience function for minimax with alpha-beta pruning.
    # This is the main function to call from external code.

    return minimax(position, depth, max_player, game, float('-inf'), float('inf'))
