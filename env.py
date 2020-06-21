import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import gym
import math
import statistics

env = gym.make("CartPole-v1")

learning_rate = 0.7
total_episodes = 8000
eps = 1
min_eps = 0.001

eps_arr = []
reward_arr = []
total_reward = 0

n_bins = (8, 16)
lower_bounds = [env.observation_space.low[2], -math.radians(50)]
upper_bounds = [env.observation_space.high[2], math.radians(50)]

print(lower_bounds)
print(upper_bounds)


def discretizer(_, __, angle, pole_velocity):
    """Convert continues state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds])
    return tuple(map(int,est.transform([[angle, pole_velocity]])[0]))


qtable = np.zeros(n_bins + (env.action_space.n,))


def policy(state: tuple, eps = 1):
    return np.argmax(qtable[state])


def return_true_action(env, current_state, eps):
    if np.random.random() < eps:
        action = env.action_space.sample()
    else:
        action = policy(current_state)
    return action


def new_Q_value( reward : float,  new_state : tuple, old, epsi,  discount_factor=1.0):
    future_optimal_value = np.max(qtable[new_state])
    learned_value = reward + discount_factor * future_optimal_value - old
    #learned_value = reward + discount_factor * (epsi * statistics.mean(qtable[new_state]) + (1 - epsi) * max(qtable[new_state])) - old
    return learned_value


for t in range(total_episodes):
    current_state, done = discretizer(*env.reset()), False
    print(t)

    if t%100 == 0:
        print(qtable)
    while not done:
        action = return_true_action(env, current_state, eps)
        obs, reward, done, _ = env.step(action)
        reward *= 0.01
        total_reward += reward
        new_state = discretizer(*obs)

        old_value = qtable[current_state][action]
        learnt_value = new_Q_value(reward, new_state, old_value, eps, 0.6)
        qtable[current_state][action] = old_value + (learning_rate * learnt_value)

        current_state = new_state

        env.render()
    eps_arr.append(eps)
    reward_arr.append(total_reward)
    if eps > min_eps:
        eps *= 0.9983
    total_reward = 0

with open('neps-arr.txt', 'w') as file:
    for i in range(len(eps_arr)):
        print(eps_arr[i], file=file)

with open('nreward-arr.txt', 'w') as file:
    for i in range(len(reward_arr)):
        print(reward_arr[i], file=file)

f=open("nq-table.txt","w")
for i in range(len(qtable)):
    for j in range(len(qtable[0])):
        print(qtable[i][j],file=f, end=" ")
    print("",file=f)
f.close()

env.reset()
rewards = []

for episode in range(100):
    current_state, done = discretizer(*env.reset()), False
    step = 0
    done = False
    total_rewards = 0
    print("****************************************************")
    print("EPISODE ", episode)

    while True:
        env.render()
        action = policy(current_state)

        new_state, reward, done, info = env.step(action)
        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            print ("Score", total_rewards)
            break
        state = new_state
env.close()
print("Score over time: " + str(sum(rewards) / 100))