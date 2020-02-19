import numpy as np
import gym
import warnings
import time
from scipy.spatial.distance import euclidean


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
		# keep track of stats
		dist_to_target = []
		steps_to_completion = []
		
		for episode in range(self.episodes):

			start = time.time()
			for step in range(self.max_steps):

				# Sample actions randomly
				action = env.action_space.sample()
				angle = np.random.randint(0, 3)
				new_state, reward, done = env.step((action, angle), play_audio=False, show_room=False)
				#print("Reward gained: ", reward)

				if done:
					end = time.time()
					steps_to_completion.append(step+1)
					print('Done! at step ', step+1)
					print('Time: ', end-start, 'seconds')
					print('Steps/second: ', float(step+1)/(end-start))

					# reset environment for new episode

					env.reset()
					print('\n-------\n NEW EPISODE:\n')
					print('New initial agent location:', env.initial_agent_loc)
					print('New target source location:', env.target_source)
					print("Dist between src and agent:", euclidean(env.initial_agent_loc, env.target_source))
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
				
				new_state, reward, done = env.step((action, 0), self.play_audio, self.show_room)
				#print("Reward gained: ", reward)

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
				new_state, reward, done = env.step((action, 0), self.play_audio, self.show_room)

				if done:
					break


class PerfectAgentORoom2:
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
		start = time.time()

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

				angle = np.random.randint(0, 3)
				new_state, reward, done = env.step((action, angle), self.play_audio, self.show_room)
				# print("Agent's state: ", self.agent_loc)
				# print("Reward gained: ", reward)

				if done:
					end = time.time()
					print('Done! at step ', step)
					print('Time: ', end-start, 'seconds')
					print('Steps/second: ', float(step)/(end-start))
					break


class HumanAgent:
	def __init__(self, target_loc, agent_loc, episodes=1, max_steps=50, converge_steps=10, step_size = None, acceptable_radius=1, play_audio=True, show_room=True):
		"""
		This class is a human agent. The moves will be played by a human player. Easy way of navigating the environment
		ourselves for testing and debugging purposes.
		Args:
			target_loc (List[int] or np.array): the location of the target in the room
			agent_loc (List[int] or np.array): the initial location of the agent in the room
			episodes (int): # of episodes to simulate
			max_steps (int): # of steps the agent can take before stopping an episode
			converge_steps (int): # of steps the perfect agent should make before rewards
			acceptable_radius (float): radius of acceptable range the agent can be in to be considered done
			step_size (float): specified step size else we programmatically assign it
		"""
		self.episodes = episodes
		self.target_loc = target_loc.copy()
		self.agent_loc = agent_loc.copy()
		self.play_audio = play_audio
		self.show_room = show_room

		self.max_steps = max_steps
		self.converge_steps = converge_steps
		self.acceptable_radius = acceptable_radius

		# Dictionary to convert 'wasd' to numbers
		self.key_to_action = {'w': 2, 'a': 0, 's': 3, 'd': 1}
		self.valid_actions = ['w', 'a', 's', 'd']
		self.valid_angles = ['0', '1', '2']

		# Finding the total distance to determine step size (total_dis / number of steps to converge)
		x_dis = abs(self.agent_loc[0] - self.target_loc[0])
		y_dis = abs(self.agent_loc[1] - self.target_loc[1])
		total_dis = x_dis + y_dis
		if step_size:
			self.step_size = step_size
		else:
			self.step_size = (total_dis) / self.converge_steps

	def fit(self, env):
		#("Enter action (wasd) followed by orientation: (012)")
		'''
		0 = Don't orient 
		1 = Orient left 15 degrees 
		2 = Orient right 15 degrees 			
		'''
		done = False
		while not done:
			action, angle = map(str, input().split())

			if action in self.valid_actions and angle in self.valid_angles:
				new_state, reward, done = env.step((self.key_to_action[action], int(angle)), self.play_audio, self.show_room)
			else:
				# Pass some dummy action
				warnings.warn("Invalid action!")
				new_state, reward, done = env.step((0, 0), self.play_audio, self.show_room)

		#("Agent's state: ", self.agent_loc)
		#print("Reward gained: ", reward)


if __name__ == '__main__':
	action, angle = map(str, input().split())
	#print(action, angle)