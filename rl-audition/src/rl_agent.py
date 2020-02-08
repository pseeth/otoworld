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
	def __init__(self, target_loc, agent_loc, episodes=1, steps=50, play_audio=True, show_room=True):
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
		self.play_audio = play_audio
		self.show_room = show_room

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

				new_state, reward, done = env.step(action, self.play_audio, self.show_room)
				print("Reward gained: ", reward)

				if done:
					break


class PerfectAgentORoom:
	def __init__(self, target_loc, agent_loc, episodes=1, steps=50, play_audio=True, show_room=True):
		"""
		This class represents a perfect agent in a non-ShoeBox environment that will randomly choose a target,
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
		self.play_audio = play_audio
		self.show_room = show_room

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
		visited = {}
		for episode in range(self.episodes):
			for step in range(self.max_steps):

				if self.agent_loc[0] < self.target_loc[0]:
					if env.check_if_inside([self.agent_loc[0] + 1, self.agent_loc[1]]):
						action = 1
						self.agent_loc[0] += 1
					elif env.check_if_inside([self.agent_loc[0], self.agent_loc[1]-1]) and (self.agent_loc[0], self.agent_loc[1] - 1) not in visited:
						action = 3
						self.agent_loc[1] -= 1
					elif env.check_if_inside([self.agent_loc[0], self.agent_loc[1] + 1]):
						action = 2
						self.agent_loc[1] += 1
				elif self.agent_loc[0] > self.target_loc[0]:
					if env.check_if_inside([self.agent_loc[0] - 1, self.agent_loc[1]]):
						action = 0
						self.agent_loc[0] -= 1
					elif env.check_if_inside([self.agent_loc[0], self.agent_loc[1] - 1]) and (self.agent_loc[0], self.agent_loc[1] - 1) not in visited:
						action = 3
						self.agent_loc[1] -= 1
						# Added to the visited list
						visited[(self.agent_loc[0], self.agent_loc[1])] = 1
					elif env.check_if_inside([self.agent_loc[0], self.agent_loc[1] + 1]):
						action = 2
						self.agent_loc[1] += 1

				elif self.agent_loc[0] == self.target_loc[0]:
					if self.agent_loc[1] < self.target_loc[1]:
						action = 2
						self.agent_loc[1] += 1
					else:
						action = 3
						self.agent_loc[1] -= 1

				new_state, reward, done = env.step(action, self.play_audio, self.show_room)
				print("Agent's state: ", self.agent_loc)
				print("Reward gained: ", reward)

				if done:
					break