#import numpy as np

class WindyGridworld():
    def __init__(self):

        self.state_space = (7,10)
        self.action_space = 4

        #self.grid = np.zeros(self.state_space)

        self.initial_state = [3,0]
        self.terminate_state = [3,8]

        self.north_wind = [0,0,0,1,1,1,2,2,1,0] # how many cells respective column will push agent north

        self.pos = None


    def step(self, action):

        self.pos[0] = min(max(self.pos[0] + self.north_wind[self.pos[0]] + action[0], 0), self.state_space[0]-1) # adding wind as well
        self.pos[1] = min(max(self.pos[1] + action[1], 0), self.state_space[1]-1)

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