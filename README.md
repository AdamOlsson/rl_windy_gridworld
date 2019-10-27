## Reinforcement Learning Part 5

Part 5 of my Reinforcement Learning (RL) series. During this series, I dwell into the field of RL by applying various methods to video games to learn and understand how an algorthm can learn to play by itself. The motivation for doing this series is simply by pure interest and to gain knowledge and experience in the field of Machine Learning.

The litterature follow throughout this series is Reinforcement Learning "An Introduction" by Ricard S. Button and Andrew G. Barto. ISBN: 9780262039246

## Temporal Difference
In this part I dive into Temporal Difference methods. More specifically, I implement the On-Policy Temporal Difference Control method Sarsa. I do this in a windy gridworld setting where the only difference to a regular gridworld setting is that in some columns there is a northbound wind pushing the agent. Temporal Difference methods estimates the state-action function during the episode, compared to previous algorithms which has to wait until an episode in complete to estimate the state-action function. This is advantageous compared to Dynamic Programing methods because there is no need for a complete model of the environment. Compared to Monte Carlo methods, Temporal Difference methods can be applied in online applications because it learns in an timestep by timestep fashion where Monte Carlo methods learns after the episode has finished. This is a critical consideration in tasks that have long episodes or not episodes at all. 

## Sarsa
The Sarsa algorithms follows the Generalized Policy Iteration (GPI) pattern but makes estimates if the state-action function after every timestep. Exploring actions are taking with a small probability &epsilon;.

Below is the policy learnt after 200 episodes. Interestingly enough, it learnt to take the action right on the fourth step where the wind pushes the agent one tile north resulting in a diagonal step. However, the optimal path would be to take the action in the fifth timestep but instead the algorithm chooses to take the up action resulting in skipping a tile. Upon longer training the algorithm does in fact turn to the optimal strategy and makes a diagonal step.
<p align="center"><figure><img src=https://github.com/AdamOlsson/rl_windy_gridworld/blob/master/img/policy_200_episodes.png></p>

Below is the state-action function learned after training for 200 episodes.
<p align="center"><figure><img src=https://github.com/AdamOlsson/rl_windy_gridworld/blob/master/img/state_action_function.png></p>
  
Below is the training progress during the 200 episode. Note that after around 4500 timesteps the inclination of the training curve does not improve much. From that point on the algorithm has learned close to the optimal policy and any large deviations from this inclination is due to the constant probability &epsilon; of taking an exploring action.
<p align="center"><figure><img src=https://github.com/AdamOlsson/rl_windy_gridworld/blob/master/img/train_progress.png></p>
