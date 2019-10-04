import numpy as np
from windy_gridworld import WindyGridworld
from collections import defaultdict


def get_q_state(Q, state, env):
    return lambda a : [Q[state, action] for action in np.arange(env.action_space)][a]

def sarsa(env, policy, epsilon=0.1, alpha=0.5, gamma=1, iterations=10):

    Q = defaultdict(float)
    
    for episode in range(iterations):
        state = env.reset()
        action = policy(state)
        
        game_over = False
        while not game_over:
            next_state, reward, game_over, info = env.step(action)
            next_action = policy(epsilon, Q, next_state, get_q_state(Q, state, env))

            Q[state, action] += alpha*(reward + gamma*Q[next_state, next_action] - Q[state, action])

            state = next_state; action = next_action
            
    return Q


def behavior_policy(epsilon, Q, state, q_state_actions):
    #TODO
    pass


def draw_Q(Q):
    #TODO
    pass

if __name__ == "__main__":
    env = WindyGridworld()
    sarsa(env, behavior_policy)