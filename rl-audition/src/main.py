import gym
import audio_room
import numpy as np
import rl_agent 
import os
import matplotlib.pyplot as plt
import room_types
import agent_wrapper

DIR_MALE = '../sounds/dry_recordings/dev/051_subset/'
DIR_FEMALE = '../sounds/dry_recordings/dev/050_subset/'

def choose_random_files():
	"""
	Function returns source random files using the directory constants. It chooses one file from the
	female recordings and one from the male recordings

	Returns:
		paths (List[str]): the paths to two wav files
	"""
	paths = []

	# pick random files from each directory
	for dir in [DIR_MALE, DIR_FEMALE]:
		files = os.listdir(dir)

		# remove non-wav files
		indexes = []
		for i, file in enumerate(files):
			if '.wav' not in file:
				indexes.append(i)
		for i in indexes:
			del files[i]

		# pick random file
		index = np.random.randint(len(files), size=1)[0]
		paths.append(os.path.join(dir, files[index]))

	return paths


def run_rl_agent():
	# paths of audio files
	paths = choose_random_files()

	# Simple shoe box environment
	room_config = [10, 10]
	corners = False

	# Un comment if polygon room is required
	# hex_room = room_types.Polygon(n=6, r=2, x_center=5, y_center=5)
	# x_points, y_points = hex_room.generate()
	# room_config = np.array([x_points, y_points])
	# corners = True

	agent_loc = np.array([3, 8])

	# Set up the gym environment
	env = gym.make('audio-room-v0', room_config=room_config, agent_loc=agent_loc, corners=corners, max_order=10,
				   step_size=1.0, direct_sources=paths, target=1, acceptable_radius=.8)
	env.add_sources()  # target is the 2nd source

	# Load the agent class
	agent = agent_wrapper.RLAgent(episodes=10, steps=1000)
	agent.fit(env)


if __name__ == '__main__':
	# Run RL Agent
	run_rl_agent()
