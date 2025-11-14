# Assets: https://techwithtim.net/wp-content/uploads/2020/09/assets.zip
import pygame
from checkers.constants import WIDTH, HEIGHT, SQUARE_SIZE, RED, WHITE
from checkers.game import Game
from minimax.algorithm import minimax
from rl.agent import RLAgent

FPS = 60

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Checkers')

def get_row_col_from_mouse(pos):
    x, y = pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col

def main():
    run = True
    clock = pygame.time.Clock()
    game = Game(WIN)

    # Load RL agent for WHITE
    rl_agent = RLAgent(WHITE, epsilon=0.0)  # I want no exploration in play
    try:
        rl_agent.load_model('rl/model.pth')
        print("Loaded RL model")
    except:
        print("No RL model found, using minimax")

    while run:
        clock.tick(FPS)

        if game.turn == WHITE:
            # Use RL agent
            action = rl_agent.get_action(game.get_board())
            if action:
                from rl.utils import apply_action
                new_board = apply_action(game.get_board(), action, WHITE)
                game.ai_move(new_board)
            else:
                # No moves
                run = False

        if game.winner() != None:
            print(game.winner())
            run = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = get_row_col_from_mouse(pos)
                game.select(row, col)

        game.update()
    
    pygame.quit()

main()