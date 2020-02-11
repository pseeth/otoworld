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
	def __init__(self, target_loc, agent_loc, episodes=1, max_steps=50, converge_steps=10, step_size = None, acceptable_radius=1, play_audio=True, show_room=True):
		"""
		This class represents a perfect agent that will randomly choose a target,
		then move throughout the room to the correct x value, then move to the correct y value and stop
		once the target is reached.

		Args:
			target_loc (List[int] or np.array): the location of the target in the room
			agent_loc (List[int] or np.array): the initial location of the agent in the room
			episodes (int): # of episodes to simulate
			max_steps (int): # of steps the agent can take before stopping an episode
			converge_steps (int): # of steps the perfect agent should make before rewards
			acceptable_radius (float): radius of acceptable range the agent can be in to be considered done
			step_size (float): specificed step size else we programmatically assign it
		"""
		self.episodes = episodes
		self.target_loc = target_loc.copy()
		self.agent_loc = agent_loc.copy()
		self.play_audio = play_audio
		self.show_room = show_room

		self.max_steps = max_steps
		self.converge_steps = converge_steps
		self.acceptable_radius = acceptable_radius

		# Finding the total distance to determine step size (total_dis / number of steps to converge)
		x_dis = abs(self.agent_loc[0] - self.target_loc[0])
		y_dis = abs(self.agent_loc[1] - self.target_loc[1])
		total_dis = x_dis + y_dis
		if step_size:
			self.step_size = step_size
		else:
			self.step_size = (total_dis) / self.converge_steps

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
				
				# Agent is to the left and outside of acceptable radius
				if self.agent_loc[0] < self.target_loc[0] and abs(self.target_loc[0] - self.agent_loc[0]) > self.acceptable_radius:
					action = 1
					self.agent_loc[0] += self.step_size
				# Agent is to the right and outside of acceptable radius
				elif self.agent_loc[0] > self.target_loc[0] and abs(self.agent_loc[0] - self.target_loc[0]) > self.acceptable_radius:
					action = 0
					self.agent_loc[0] -= self.step_size
				# Agent is to the above and outside of acceptable radius
				elif self.agent_loc[1] > self.target_loc[1] and abs(self.agent_loc[1] - self.target_loc[1]) > self.acceptable_radius:
					action = 3
					self.agent_loc[1] -= self.step_size
				# Agent is to the below and outside of acceptable radius
				elif self.agent_loc[1] < self.target_loc[1] and abs(self.agent_loc[1] - self.target_loc[1]) > self.acceptable_radius:
					action = 2
					self.agent_loc[1] += self.step_size
				# We are in the range that we want but we want to move a little bit
				else:
					# TODO: figure out how to discount the step size once we are in this cycle
					self.step_size = self.acceptable_radius / 2
					env.step_size = self.step_size
					x_dis = self.target_loc[0] - self.agent_loc[0]
					y_dis = self.target_loc[1] - self.agent_loc[1]
					if abs(x_dis) > abs(y_dis):
						if x_dis < 0:
							action = 0
							self.agent_loc[0] -= self.step_size
						else:
							action = 1
							self.agent_loc += self.step_size
					else:
						if y_dis < 0:
							action = 3
							self.agent_loc[1] -= self.step_size
						else:
							action = 2
							self.agent_loc += self.step_size
				
				new_state, reward, done = env.step(action, self.play_audio, self.show_room)
				print("Reward gained: ", reward)

				if done:
					break


class PerfectAgentORoom:
	def __init__(self, target_loc, agent_loc, episodes=1, max_steps=50, converge_steps=10, step_size=None, acceptable_radius=1, play_audio=True, show_room=True):
		"""
		This class represents a perfect agent in a non-ShoeBox environment that will randomly choose a target,
		then move throughout the room to the correct x value, then move to the correct y value and stop
		once the target is reached.

		Args:
			target_loc (List[int] or np.array): the location of the target in the room
			agent_loc (List[int] or np.array): the initial location of the agent in the room
			episodes (int): # of episodes to simulate
			max_steps (int): # of steps the agent can take before stopping an episode
			converge_steps (int): # of steps the perfect agent should make before rewards
			acceptable_radius (float): radius of acceptable range the agent can be in to be considered done
			step_size (float): specificed step size else we programmatically assign it
		"""
		self.episodes = episodes
		self.max_steps = max_steps
		self.target_loc = target_loc.copy()
		self.agent_loc = agent_loc.copy()
		self.play_audio = play_audio
		self.show_room = show_room
		self.converge_steps = converge_steps
		self.acceptable_radius = acceptable_radius

		# Finding the total distance to determine step size (total_dis / number of steps to converge)
		x_dis = abs(self.agent_loc[0] - self.target_loc[0])
		y_dis = abs(self.agent_loc[1] - self.target_loc[1])
		total_dis = x_dis + y_dis
		if step_size:
			self.step_size = step_size
		else:
			self.step_size = (total_dis) / self.converge_steps

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
		visited = set()
		for episode in range(self.episodes):
			for step in range(self.max_steps):

				# Agent is to the left and outside of acceptable radius
				if self.agent_loc[0] < self.target_loc[0] and (self.target_loc[0] - self.agent_loc[0]) > self.acceptable_radius:
					# Check if the agent will be inside, if outside, try moving up or down
					if env.check_if_inside([self.agent_loc[0] + self.step_size, self.agent_loc[1]]):
						action = 1
						self.agent_loc[0] += self.step_size
					elif env.check_if_inside([self.agent_loc[0], self.agent_loc[1]-self.step_size]) and (self.agent_loc[0], self.agent_loc[1] - self.step_size) not in visited:
						action = 3
						self.agent_loc[1] -= self.step_size
					elif env.check_if_inside([self.agent_loc[0], self.agent_loc[1] + self.step_size]):
						action = 2
						self.agent_loc[1] += self.step_size
				
				# Agent is to the right and outside of acceptable radius
				elif self.agent_loc[0] > self.target_loc[0] and (self.agent_loc[0] - self.target_loc[0]) > self.acceptable_radius:
					# Check if the agent will be inside, if outside, move up or down
					if env.check_if_inside([self.agent_loc[0] - self.step_size, self.agent_loc[1]]):
						action = 0
						self.agent_loc[0] -= self.step_size
					elif env.check_if_inside([self.agent_loc[0], self.agent_loc[1] - self.step_size]) and (self.agent_loc[0], self.agent_loc[1] - self.step_size) not in visited:
						action = 3
						self.agent_loc[1] -= self.step_size
					elif env.check_if_inside([self.agent_loc[0], self.agent_loc[1] + self.step_size]):
						action = 2
						self.agent_loc[1] += self.step_size

				# Agent is to the above and outside of acceptable radius
				elif self.agent_loc[1] > self.target_loc[1] and (self.agent_loc[1] - self.target_loc[1]) > self.acceptable_radius:
					# Check if the agent will be inside, if outside, move right or left
					if env.check_if_inside([self.agent_loc[0], self.agent_loc[1] - self.step_size]):
						action = 3
						self.agent_loc[1] -= self.step_size
					elif env.check_if_inside([self.agent_loc[0] + self.step_size, self.agent_loc[1]]) and (self.agent_loc[0] + self.step_size, self.agent_loc[1]) not in visited:
						action = 1
						self.agent_loc[0] += self.step_size
					elif env.check_if_inside([self.agent_loc[0] - self.step_size, self.agent_loc[1]]):
						action = 0
						self.agent_loc[0] -= self.step_size
				
				# Agent is to the below and outside of acceptable radius
				elif self.agent_loc[1] < self.target_loc[1] and (self.target_loc[1] - self.agent_loc[1]) > self.acceptable_radius:
					if env.check_if_inside([self.agent_loc[0], self.agent_loc[1] - self.step_size]):
						action = 2
						self.agent_loc[1] += self.step_size
					elif env.check_if_inside([self.agent_loc[0] + self.step_size, self.agent_loc[1]]) and (self.agent_loc[0] + self.step_size, self.agent_loc[1]) not in visited:
						action = 1
						self.agent_loc[0] += self.step_size
					elif env.check_if_inside([self.agent_loc[0] - self.step_size, self.agent_loc[1]]):
						action = 0
						self.agent_loc[0] -= self.step_size
				else:
					# We are in the range that we want but we want to move a little bit
					# TODO: figure out how to discount the step size once we are in this cycle
					self.step_size = self.acceptable_radius / 2
					env.step_size = self.step_size
					x_dis = self.target_loc[0] - self.agent_loc[0]
					y_dis = self.target_loc[1] - self.agent_loc[1]
					if abs(x_dis) > abs(y_dis):
						if x_dis < 0:
							action = 0
							self.agent_loc[0] -= self.step_size
						else:
							action = 1
							self.agent_loc += self.step_size
					else:
						if y_dis < 0:
							action = 3
							self.agent_loc[1] -= self.step_size
						else:
							action = 2
							self.agent_loc += self.step_size
				
				visited.add((self.agent_loc[0], self.agent_loc[1]))
				new_state, reward, done = env.step(action, self.play_audio, self.show_room)

				if done:
					break
