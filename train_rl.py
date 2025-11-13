import torch
from checkers.board import Board
from checkers.game import Game
from checkers.constants import RED, WHITE
from minimax.algorithm import minimax
from rl.agent import RLAgent
from rl.utils import board_to_state, apply_action
import pygame

# Initialize pygame for board operations, but no display
pygame.init()
WIN = pygame.display.set_mode((1,1))  # Dummy window

def play_game(agent, opponent_color, max_moves=100):
    game = Game(WIN)
    board = game.get_board()
    done = False
    moves = 0
    while not done and moves < max_moves:
        if game.turn == agent.color:
            state = board_to_state(board)
            action = agent.get_action(board)
            if action is None:
                reward = -1
                next_state = state
                done = True
            else:
                next_board = apply_action(board, action, agent.color)
                next_state = board_to_state(next_board)
                game.board = next_board
                game.change_turn()
                board = next_board
                reward = 0
                if game.winner() != None:
                    if game.winner() == agent.color:
                        reward = 1
                    else:
                        reward = -1
                    done = True
            agent.remember(state, action, reward, next_state, done)
        else:
            # opponent
            is_max_player = (opponent_color == WHITE)
            value, new_board = minimax(board, 2, is_max_player, game)
            game.ai_move(new_board)
            board = new_board
            if game.winner() != None:
                done = True
        moves += 1
    return done and reward if done else 0

def train_agent(episodes=100):
    agent = RLAgent(WHITE)  # Agent plays as WHITE
    for episode in range(episodes):
        reward = play_game(agent, RED)
        agent.replay()
        if episode % 10 == 0:
            print(f"Episode {episode}, Epsilon: {agent.epsilon:.3f}, Reward: {reward}")
    agent.save_model('rl/model.pth')
    print("Training complete")

if __name__ == "__main__":
    train_agent()