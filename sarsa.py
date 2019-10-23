import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from windy_gridworld import WindyGridworld
from collections import defaultdict

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
            next_action = policy(epsilon, env.get_actions(next_state), get_q_state(Q, state, env))

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

def get_policy(Q, shape):
    greedy_actions = np.zeros(shape)

    for (y,x), a in Q.keys():
        if greedy_actions[y,x] == 0 or greedy_actions[y,x] < Q[(y,x), a] and not Q[(y,x), a] == 0:
            greedy_actions[y,x] = a
    
    return greedy_actions

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

    plt.show()
    

# TODO: Verify this and plot policy
def play(env, policy):
    history = []

    state = env.reset()
    game_over = False

    while not game_over:
        history.append(state)
        action = policy[state[0], state[1]]
        next_state, reward, game_over, info = env.step(action)
        state = next_state

    return history

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
    Q = sarsa(env, behavior_policy, iterations=10)

    draw_Q(Q, env.state_space)