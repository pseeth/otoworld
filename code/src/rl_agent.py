import numpy as np
import gym


class RandomAgent(object):
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


class PerfectAgent(object):
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


class PerfectAgentHorsehoeRoom(object):
	def __init__(self, target_loc, agent_loc, episodes=1, steps=50):
		'''
		Room shape:
			| |		| |
			| |_____| |
			|_________|
		:param target_loc:
		:param agent_loc:
		:param episodes:
		:param steps:
		'''
		self.episodes = episodes
		self.max_steps = steps
		self.target_loc = target_loc.copy()
		self.agent_loc = agent_loc.copy()


	def fit(self, env):
		'''
		Hardcoding the path to target for now
		Also, remember
		0 = Left
		1 = Right
		2 = Up
		3 = Down

		Args:
		:param env (OpenAI Gym Env): the gym environment class
		'''
		for episode in range(self.episodes):
			for step in range(self.max_steps):
				# NOTE: hardcoded (go down to 3, right to 9, up to 8) to get from [3,8] to [9,8]
				while (self.agent_loc[0] != self.target_loc[0]) or (self.agent_loc[1] != self.target_loc[1]):
					# need to go down to 3
					if self.agent_loc[0] == 2:
						action = 3
						self.agent_loc[1] -= 1

						# if we reach 3, stop going down
						if self.agent_loc[1] == 3:
							break

					# go right to 9
					elif self.agent_loc[1] == 3:
						action = 1
						self.agent_loc[0] += 1

						if self.agent_loc[0] == 9:
							break

					# go up to 8
					else:
						action = 2
						self.agent_loc[1] += 1

						if self.agent_loc[1] == 8:
							break

				new_state, reward, done = env.step(action)
				print("Reward gained: ", reward)

				if done:
					break
