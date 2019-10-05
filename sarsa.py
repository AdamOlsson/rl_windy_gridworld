import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from windy_gridworld import WindyGridworld
from collections import defaultdict

ACT_TO_IND = {[1,0]:0, [-1,0]:1, [0,1]:2, [0,-1]:3}
IND_TO_ACT = {0:[1,0], 1:[-1,0], 2:[0,1], 3:[0,-1]}

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
            next_action = policy(epsilon, env.get_actions(state), get_q_state(Q, state, env))

            Q[state, action] += alpha*(reward + gamma*Q[next_state, next_action] - Q[state, action])

            state = next_state; action = next_action
            
    return Q

# TODO
def behavior_policy(epsilon, actions, q_state_values):

    action_id = [ACT_TO_IND[a] for a in actions]            # index of actions
    state_values = [q_state_values(i) for i in action_id]   # state value of actions

    # TODO: Find best action

    if np.random.random_sample() < epsilon:
        # Take a random action from the set of non-greedy actions
        pass
    else:
        # Take a greedy action
        pass


def draw_Q(Q):
    #TODO
    pass

def animate_Q(env, Q):
    # TODO

    grid = np.zeros(env.state_space)

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-1, 11), ylim=(-1, 8))

    ax.imshow(grid) # testing
    
    agent = ax.plot([],[], bo=3, ms=6)

    def init():
        env.reset()
        agent.set_data([],[])
        return agent
    
    def animate(i):
        #action = greedy_action(state) # TODO
        action = None
        state = env.step(action)

        agent.set_data(state[0], state[1])

        return agent

    anim = FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
    plt.show()

if __name__ == "__main__":
    env = WindyGridworld()
    sarsa(env, behavior_policy)
