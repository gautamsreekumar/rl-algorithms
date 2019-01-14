import math
from collections import defaultdict
import gym
import numpy as np
from gym import spaces
import numpy.random as nprand
from gym.envs.classic_control import rendering
import time

class FourRooms(gym.Env):

	UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
	metadata = {'render.modes': ['human']}

	def __init__(self, l=32, f=30, west=1, neg=1):
		self.action_count = 4
		self.scaling_factor = 10
		self.viewer = None
		self.neg = neg
		self.l = l # length of the grid world
		# in this case, l=12
		self.transition_reward = -1
		self.west = west
		# no penalty for each step
		self.reward_range = [self.transition_reward, self.transition_reward]
		self.starting_states = [0]
		self.state = nprand.choice(a=self.starting_states) # starting states
		self.finish = [f]
		self.p = 0.9
		self.p_ = (1-self.p)/(self.action_count-1)
		self.action_space = spaces.Discrete(4)
		self.b_states_1 = [self.l*_ + 12 for _ in range(self.l) if _ != self.l-7]
		self.b_states_2 = [self.l*14+_ for _ in range(12) if _ != 5]
		self.b_states_3 = [self.l*19+_ for _ in range(12, self.l) if _ != 25]
		self.observation_space = spaces.Discrete(self.l**2)
		self._state_transition_prob = self.__generate_state_transition_prob__()
		self.initialize_grid(f)

	def _step(self, action):
		assert self.action_space.contains(action)
		self.state = self.next_state(self.state, action)
		observation = {'state': self.state}
		reward = self._get_reward()
		done = self.state in self.finish
		info = None
		return observation, reward, done, info

	def _reset(self):
		self.state = nprand.choice(a=self.starting_states) # starting states
		return self.state

	def _render(self, mode='human', close=False):
		factor = self.scaling_factor # scaling factor; should be a multiple of 10
		window_width = self.l*factor
		window_height = self.l*factor
		if self.viewer is None:
			self.viewer = rendering.Viewer(window_width, window_height)
			l, r, t, b = 0, window_width, window_height/2, -window_height/2
			outline = rendering.make_polygon([(l,b), (l,t), (r,t), (r,b)], filled=False)
			for i in np.arange(start=1*factor, stop=(self.l-1)*factor+1, step=factor):
				line_h = rendering.Line((0, i), (window_width, i))
				line_w = rendering.Line((i, 0), (i, window_height))
				self.viewer.add_geom(line_w)
				self.viewer.add_geom(line_h)

			for i in self.b_states_1:
				partition1 = rendering.make_polygon(self.c(i))
				partition1.set_color(0.1, 0.1, 0.1)
				self.viewer.add_geom(partition1)

			for i in self.b_states_2:
				partition2 = rendering.make_polygon(self.c(i))
				partition2.set_color(0.1, 0.1, 0.1)
				self.viewer.add_geom(partition2)

			for i in self.b_states_3:
				partition3 = rendering.make_polygon(self.c(i))
				partition3.set_color(0.1, 0.1, 0.1)
				self.viewer.add_geom(partition3)
			
			agent_pos = self.state
			agent_x = agent_pos % self.l
			agent_y = math.floor(agent_pos / self.l)
			agent_coords = [(agent_x*factor+10, agent_y*factor+10),
							((agent_x+1)*factor-10, agent_y*factor+10),
							((agent_x+1)*factor-10, (agent_y+1)*factor-10),
							(agent_x*factor+10, (agent_y+1)*factor-10)]
			agent_point = rendering.make_polygon(agent_coords)
			agent_point.set_color(0.5, 0, 0.5)
			self.viewer.add_onetime(agent_point)
			goal = self.c(self.finish[0])
			goal_point = rendering.make_polygon(goal)
			goal_point.set_color(0, 0, 1)
			self.viewer.add_geom(goal_point)
			for i in self.starting_states:
				start = self.c(i)
				start_point = rendering.make_polygon(start)
				start_point.set_color(0, 0.5, 0)
				self.viewer.add_geom(start_point)

		agent_pos = self.state
		agent_x = agent_pos % self.l
		agent_y = math.floor(agent_pos / self.l)
		agent_coords = [(agent_x*factor+10, agent_y*factor+10),
						((agent_x+1)*factor-10, agent_y*factor+10),
						((agent_x+1)*factor-10, (agent_y+1)*factor-10),
						(agent_x*factor+10, (agent_y+1)*factor-10)]
		agent_point = rendering.make_polygon(agent_coords)
		agent_point.set_color(0.5, 0, 0.5)
		self.viewer.add_onetime(agent_point)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def _seed(self, seed=None):
		pass

	# Control methods

	def _get_reward(self):
		curr_state = self.state
		if curr_state in self.finish:
			return +10
		else:
			return -0.5*self.neg

	def next_state(self, state, action):
		assert self.observation_space.contains(state)
		assert self.action_space.contains(action)
		agent_row = math.floor(state / self.l)
		agent_col = state % self.l

		# Handle Up
		if action == self.UP:
			if agent_row == 0:
				next_state = state
			else:
				next_state = state-self.l
		# Handle Right
		if action == self.RIGHT:
			if agent_col == self.l-1:
				next_state = state
			else:
				next_state = state+1
		# Handle Down
		if action == self.DOWN:
			if agent_row == self.l-1:
				next_state = state
			else:
				next_state = state+self.l
		# Handle Left
		if action == self.LEFT:
			if agent_col == 0:
				next_state = state
			else:
				next_state = state-1

		if next_state in self.b_states_1 or next_state in self.b_states_2 or next_state in self.b_states_3:
			return state
		else:
			return next_state

	# Additional functionality

	def initialize_grid(self, f):
		self.grid = np.zeros([self.l, self.l]) # initializing grid
		self.grid[f//self.l, f%self.l] = +10 # goal is to reach A

	def c(self, state):
		factor = self.scaling_factor
		x = state % self.l
		y = math.floor(state/self.l)
		coord1 = (x*factor, y*factor)
		coord2 = (x*factor, (y+1)*factor)
		coord3 = ((x+1)*factor, (y+1)*factor)
		coord4 = ((x+1)*factor, y*factor)
		coords = [coord1, coord2, coord3, coord4]
		return coords

	def state_transition_prob(self, s1, r, s, a):
		return self._state_transition_prob.get((s1, r, s, a), 0)

	def __generate_state_transition_prob__(self):
		prob = defaultdict()

		for s in range(1, self.observation_space.n - 1):
			for a in range(self.action_space.n):
				s1 = self.next_state(s, a)
				prob[(s1, self.transition_reward, s, a)] = 1.0
		return prob