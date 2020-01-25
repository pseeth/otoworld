import numpy as np
import gym


class RandomAgent:
	def __init__(self, episodes=1, steps=10):
		self.episodes = episodes
		self.max_steps = steps

	def fit(self, env):
		for episode in range(self.episodes):
			for step in range(self.max_steps):

				# Sample actions randomly
				action = env.action_space.sample()
				new_state, reward, done = env.step(action)
				print("Reward gained: ", reward)

				if done:
					break


class PerfectAgent:
	def __init__(self, target_loc, agent_loc, episodes=1, steps=50):
		self.episodes = episodes
		self.max_steps = steps
		self.target_loc = target_loc.copy()
		self.agent_loc = agent_loc.copy()

	def fit(self, env):
		'''
		Strategy to always reach goal.
		1. Reduce the distance in x direction
		2. Reduce the distance in y direction
		Also, remember
		0 = Left
		1 = Right
		2 = Up
		3 = Down
		'''

		for episode in range(self.episodes):
			for step in range(self.max_steps):

				if self.agent_loc[0] < self.target_loc[0]:
					action = 1
					self.agent_loc[0] += 1
				elif self.agent_loc[0] > self.target_loc[0]:
					action = 0
					self.agent_loc[0] -= 1
				elif self.agent_loc[0] == self.target_loc[0]:
					if self.agent_loc[1] < self.target_loc[1]:
						action = 2
						self.agent_loc[1] += 1
					else:
						action = 3
						self.agent_loc[1] -= 1

				new_state, reward, done = env.step(action)
				print("Reward gained: ", reward)

				if done:
					break



