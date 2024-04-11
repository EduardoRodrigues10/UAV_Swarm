import numpy as np
import random

class QLearning:
    def __init__(self, nb_rows, nb_cols, alpha, gamma, epsilon):
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.nb_actions = 4
        self.FORWARD, self.BACKWARD, self.LEFT, self.RIGHT = range(self.nb_actions)
        self.OUT, self.EMPTY, self.INTEREST = -45, -5, 50
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((nb_rows * nb_cols, self.nb_actions))

    def state_to_coordinates(self, state):
        return state % self.nb_cols, state // self.nb_rows

    def coordinates_to_state(self, x, y):
        return y * self.nb_rows + x

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.nb_actions)
        else:
            return np.argmax(self.Q[state])

    def do_action(self, state, action, grid):
        x, y = self.state_to_coordinates(state)

        if action == self.FORWARD:
            if y == 0:
                return state, self.OUT, False
            else:
                if grid[y - 1][x] == 1:
                    return self.coordinates_to_state(x, y - 1), self.INTEREST, True
                else :
                    return self.coordinates_to_state(x, y - 1), self.EMPTY, False
        elif action == self.BACKWARD:
            if y == 2:
                return state, self.OUT, False
            else:
                if grid[y + 1][x] == 1:
                    return self.coordinates_to_state(x, y + 1), self.INTEREST, True
                else :
                    return self.coordinates_to_state(x, y + 1), self.EMPTY, False
        elif action == self.LEFT:
            if x == 0:
                return state, self.OUT, False
            else:
                if grid[y][x - 1] == 1:
                    return self.coordinates_to_state(x - 1, y), self.INTEREST, True
                else :
                    return self.coordinates_to_state(x - 1, y), self.EMPTY, False
        elif action == self.RIGHT:
            if x == 2:
                return state, self.OUT, False
            else:
                if grid[y][x + 1] == 1:
                    return self.coordinates_to_state(x + 1, y), self.INTEREST, True
                else :
                    return self.coordinates_to_state(x + 1, y), self.EMPTY, False

    def train(self, episodes, grids):
        for _ in range(episodes):
            done = 0
            state = 0
            rewards = 0
            while done != len(grids):
                
                action = self.select_action(state)
                
                (new_state, reward, end) = self.do_action(state, action, grids[done])
                self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[new_state]) - self.Q[state][action])
                
                state = new_state
                rewards += reward

                if end:
                    done += 1

    def get_actions(self, grids):
        done = 0
        rewards = 0
        state = 0
        actions = []
        while done != len(grids):
            best_action = np.argmax(self.Q[state])
            actions.append(best_action)
            new_state, reward, end = self.do_action(state, best_action, grids[done])
            
            rewards += reward
            state = new_state
            if end:
                done += 1

        return actions