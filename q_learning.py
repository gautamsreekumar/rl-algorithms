import gym
import rooms
import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt
import time
import os

def get_policy(Q, eps, n_a):
	policy = np.ones(n_a)*(eps/n_a)
	qmax_index = np.argmax(Q)
	policy[qmax_index] += 1-eps
	return policy

env = gym.make('FourRooms-v0')
env.seed(1)
start_time = time.time()

total_episodes = 10000
reward_per_episode = 0
plot_dir = './plots_q_learning/'
if not os.path.exists(plot_dir):
	os.makedirs(plot_dir)

# env details
grid_s = 32

total_reward = np.zeros(total_episodes)
avg_rew_of_episode = np.zeros(total_episodes)
episode_length = np.zeros(total_episodes)
gamma = 0.9
alpha = 0.1
n_a = 4 # number of available actions
n_states = grid_s*grid_s
Q = nprand.normal(size=[n_states, n_a]) # initializing Q-estimates
eps = 0.01 # epsilon for generating epsilon-greedy policy (target)
runs = 20

for run in range(runs):
	Q = nprand.normal(size=[n_states, n_a]) # initializing Q-estimates
	eps = 0.4
	max_steps = 1000
	for i in range(total_episodes):
		state = env.reset()
		end_of_episode = False
		t = 0
		reward_so_far = 0
		num_steps = 0
		while not end_of_episode and num_steps < max_steps:
			t += 1
			policy = get_policy(Q[state], eps, n_a)
			# env.render()
			move = nprand.choice(a=4, p=policy)
			next_state, reward, end_of_episode, __ = env.step(move)
			next_state = next_state['state']
			reward_so_far += reward
			episode_length[i] += 1
			avg_rew_of_episode[i] += reward
			next_action = np.argmax(Q[next_state])
			Q[state,move] = Q[state,move] + alpha*(reward+gamma*Q[next_state,next_action] - Q[state,move])
			state = next_state
			num_steps += 1
		print "Time {}s Run {} Episode {} Reward {}".format(time.time()-start_time, run, i, reward_so_far)
		eps = eps*0.995

	temp_avg_rew_of_episode = avg_rew_of_episode/float(run)
	plt.plot(temp_avg_rew_of_episode)
	plt.xlabel('Episode')
	plt.ylabel('Average reward of each episode')
	plt.savefig(plot_dir+'avg_rew_of_episode.png')
	# plt.show()
	plt.clf()

	temp_episode_length = episode_length/float(run)
	plt.plot(temp_episode_length)
	plt.xlabel('Episode')
	plt.ylabel('Average episode length')
	plt.savefig(plot_dir+'avg_episode_length.png')
	# plt.show()
	plt.clf()

# generating plots

avg_rew_of_episode = avg_rew_of_episode/(1.0*runs)
episode_length = np.array(episode_length)/(1.0*runs)
np.save(plot_dir+'avg_rew_of_episode', avg_rew_of_episode)
np.save(plot_dir+'episode_length', episode_length)
np.save(plot_dir+'q_values', Q)

plt.plot(avg_rew_of_episode)
plt.grid()
plt.xlabel('Episode')
plt.ylabel('Average reward of each episode')
plt.savefig(plot_dir+'avg_rew_of_episode.png')
# plt.show()
plt.clf()

plt.plot(episode_length)
plt.grid()
plt.xlabel('Episode')
plt.ylabel('Average episode length')
plt.savefig(plot_dir+'avg_episode_length.png')
# plt.show()