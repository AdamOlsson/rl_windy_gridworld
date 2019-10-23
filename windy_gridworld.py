#import numpy as np

class WindyGridworld():
    def __init__(self):

        self.state_space = (7,10)
        self.action_space = 4

        self.action_list = [(1,0), (-1,0), (0,1), (0,-1)]

        #self.grid = np.zeros(self.state_space)

        self.initial_state = (3,0)
        self.terminate_state = (3,7)

        self.north_wind = [0,0,0,-1,-1,-1,-2,-2,-1,0] # how many cells respective column will push agent north

        self.pos = None


    def step(self, action):

        pos0 = min(max(self.pos[0] + self.north_wind[self.pos[1]] + action[0], 0), self.state_space[0]-1) # adding wind as well
        pos1 = min(max(self.pos[1] + action[1], 0), self.state_space[1]-1)

        self.pos = (pos0, pos1)

        if self.pos == self.terminate_state:
            game_over = True
            reward = 0
        else:
            game_over = False
            reward = -1

        info = []
        
        return self.pos, reward, game_over, info


    def reset(self):
        self.pos = self.initial_state

        return self.pos

    # Returns the set of available action for a given state
    def get_actions(self, state):
        actions = []

        if 0 <= state[0] -1:
            actions.append((-1,0))
        if state[0] +1 < self.state_space[0]:
            actions.append((1,0))
        if 0 <= state[1] -1:
            actions.append((0,-1))
        if state[1] +1 < self.state_space[1]:
            actions.append((0,1))
        
        return actions