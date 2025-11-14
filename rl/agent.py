import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .model import ValueNetwork
from .utils import board_to_state, get_valid_actions, apply_action
from checkers.constants import ROWS, COLS

class RLAgent:
    def __init__(self, color, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.95, gamma=0.99, lr=0.001, batch_size=64, memory_size=10000):
        self.color = color
        self.input_size = ROWS * COLS * 5
        self.model = ValueNetwork(self.input_size)
        self.target_model = ValueNetwork(self.input_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = 1000
        self.step_count = 0

    def get_action(self, board):
        valid_actions = get_valid_actions(board, self.color)
        if not valid_actions:
            return None
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            # Evaluate each possible next state
            best_action = None
            best_value = -float('inf')
            for action in valid_actions:
                next_board = apply_action(board, action, self.color)
                next_state = board_to_state(next_board)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                with torch.no_grad():
                    value = self.model(next_state_tensor).item()
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        current_values = self.model(states).squeeze()
        next_values = self.target_model(next_states).squeeze()

        targets = rewards + self.gamma * next_values * (1 - dones)

        loss = F.mse_loss(current_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_count % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_count += 1

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(torch.load(path))