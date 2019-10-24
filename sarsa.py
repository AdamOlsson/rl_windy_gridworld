import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from windy_gridworld import WindyGridworld
from collections import defaultdict
import pickle

ACT_TO_IND = {(1,0):0, (-1,0):1, (0,1):2, (0,-1):3}
IND_TO_ACT = {0:(1,0), 1:(-1,0), 2:(0,1), 3:(0,-1)}

def get_q_state(Q, state, env):
    return lambda a : [Q[state,action] for action in np.arange(env.action_space)][a]

'''Function only handles reading from Q, not writing. Removes the need to call ACT_TO_IND[x] on every Q read'''
def create_q_reader(Q):
    return lambda s, a : Q[s, ACT_TO_IND[a]]


def sarsa(env, policy, epsilon=0.1, alpha=0.5, gamma=1, iterations=1000):

    Q = defaultdict(float)
    Q_reader = create_q_reader(Q)
    no_steps = 0

    for episode in range(iterations):
        if episode % 10 == 0:
            print("Playing episode {} out of {}. Average number of steps: {}.".format(episode, iterations, no_steps/10))
            no_steps = 0

        state = env.reset()
        action = policy(epsilon, env.get_actions(state), get_q_state(Q, state, env))

        game_over = False
        while not game_over:
            next_state, reward, game_over, info = env.step(action)
            next_action = policy(epsilon, env.get_actions(next_state), get_q_state(Q, next_state, env))

            Q[state, ACT_TO_IND[action]] += alpha*(reward + gamma*Q_reader(next_state, next_action) - Q_reader(state, action))

            state = next_state; action = next_action
            no_steps += 1
            
    return Q


def behavior_policy(epsilon, actions, q_state_values):
    '''
    Select the a greedy action with prob 1-epsilon. Select a non-greedy action with prob epsilon
    '''
    action_ids = [ACT_TO_IND[a] for a in actions]            # index of actions
    state_values = np.array([q_state_values(i) for i in action_ids])   # state value of actions

    best_action     = np.random.choice([action_ids[i] for i in np.where(state_values == state_values.max())[0]]) # select all greedy actions
    random_action   = np.random.choice([action_ids[i] for i in np.where(action_ids != best_action)[0]]) # select non-greedy action

    # Explore or greedy action?
    action_id = np.random.choice([best_action, random_action], p=[1-epsilon, epsilon])

    return IND_TO_ACT[action_id]


def draw_Q(Q, shape):

    best_q_values = np.zeros(shape)

    for (y,x), a in Q.keys():
        if best_q_values[y,x] == 0 or best_q_values[y,x] < Q[(y,x), a] and not Q[(y,x), a] == 0:
            best_q_values[y,x] = Q[(y,x), a]
    
    fig, ax = plt.subplots(figsize=(20,15))
    plt.title('State-Action Function', fontsize=30)

    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    im = ax.imshow(best_q_values)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel('Value', rotation=-90, va='bottom', fontsize=20)

    return best_q_values
    

def play_no_explore(env, policy, Q):
    history = []

    state = env.reset()
    action = policy(0, env.get_actions(state), get_q_state(Q, state, env))

    game_over = False
    while not game_over:
        history.append((state, action))

        next_state, _, game_over, _ = env.step(action)
        next_action = policy(0, env.get_actions(next_state), get_q_state(Q, next_state, env))

        state = next_state; action = next_action

    return history


#TODO Make plot nice
def plot_history(env, h):
    
    xs = []; ys = []; us = []; vs = []
    for t in range(1,len(h)-1):
        state = h[t][0]
        next_state = h[t+1][0]
        
        x = state[1]
        y = state[0]
        u = next_state[1] - state[1]
        v = -(next_state[0] - state[0])

        xs.append(x)
        ys.append(y)
        us.append(u)
        vs.append(v)


    fig = plt.figure(figsize=(20,15))
    plt.imshow(np.zeros(env.state_space))
    plt.quiver(xs, ys, us, vs)


if __name__ == "__main__":
    env = WindyGridworld()

    if False: # Save time but using pre-trained Q values
        Q = sarsa(env, behavior_policy, iterations=10000)

        with open('Q.p', 'wb') as f:
            pickle.dump(Q, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Q.p', 'rb') as f:
        Q = pickle.load(f)

    draw_Q(Q, env.state_space)

    history = play_no_explore(env, behavior_policy, Q)
    plot_history(env, history)

    plt.show()
    