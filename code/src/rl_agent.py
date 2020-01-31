import numpy as np
import gym


class RandomAgent(object):
	def __init__(self, episodes=1, steps=10):
		"""
		This class represents a random agent that will move throughout the room

		Args:
			episodes (int): # of episodes to simulate
			steps (int): # of steps the agent can take before stopping an episode
		"""
		self.episodes = episodes
		self.max_steps = steps

	def fit(self, env):
		"""
		This function runs the simulation

		Args:
			env (Gym env obj): the environment used to sample the action space randomly (0, 1, 2, 3)
		"""
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
		"""
		This class represents a perfect agent that will randomly choose a target,
		then move throughout the room to the correct x value, then move to the correct y value and stop
		once the target is reached.

		Args:
			target_loc (List[int] or np.array): the location of the target in the room
			agent_loc (List[int] or np.array): the initial location of the agent in the room
			episodes (int): # of episodes to simulate
			steps (int): # of steps the agent can take before stopping an episode
		"""
		self.episodes = episodes
		self.max_steps = steps
		self.target_loc = target_loc.copy()
		self.agent_loc = agent_loc.copy()

	def fit(self, env):
		"""
		Strategy to always reach goal.
			1. Reduce the distance in x direction
			2. Reduce the distance in y direction

		Also, remember
		0 = Left
		1 = Right
		2 = Up
		3 = Down

		Args:
			env (Gym env obj): the environment used to take the action
		"""
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


class PerfectAgentHorseshoeRoom(object):
	def __init__(self, target_loc, agent_loc, episodes=1, steps=50):
		"""
		Room shape:
			|A|		|T|
			| |_____| |
			|_________|

		Args:
			target_loc (List[int] or np.array): the location of the target in the room
			agent_loc (List[int] or np.array): the initial location of the agent in the room
			episodes (int): # of episodes to simulate
			steps (int): # of steps the agent can take before stopping an episode
		"""
		self.episodes = episodes
		self.max_steps = steps
		self.target_loc = target_loc.copy()
		self.agent_loc = agent_loc.copy()


	def fit(self, env):
		"""
		Hardcoded to reach the target. Go down to [2,3], right to [9, 3], up to [9,8] to get from [2,8] to [9,8]

		Also, remember
		0 = Left
		1 = Right
		2 = Up
		3 = Down

		Args:
			env (Gym env obj): the environment used to take the action
		"""
		for episode in range(self.episodes):
			for step in range(self.max_steps):
				while (self.agent_loc[0] != self.target_loc[0]) or (self.agent_loc[1] != self.target_loc[1]):
					# need to go down to [2,3]
					if self.agent_loc[0] == 2:
						action = 3
						self.agent_loc[1] -= 1

						# if we reach 3, stop going down
						if self.agent_loc[1] == 3:
							break

					# go right to [9,3]
					elif self.agent_loc[1] == 3:
						action = 1
						self.agent_loc[0] += 1

						if self.agent_loc[0] == 9:
							break

					# go up to [9,8]
					else:
						action = 2
						self.agent_loc[1] += 1

						if self.agent_loc[1] == 8:
							break

				new_state, reward, done = env.step(action)
				print("Reward gained: ", reward)

				if done:
					break
