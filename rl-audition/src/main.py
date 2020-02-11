import gym
import audio_room
import numpy as np
import rl_agent
import os
import matplotlib.pyplot as plt

DIR_MALE = '../sounds/dry_recordings/dev/051_subset/'
DIR_FEMALE = '../sounds/dry_recordings/dev/050_subset/'

def run_random_agent():
	"""
	This function runs the Random Agent in the ShoeBox audio environment. The agent randomly moves around until it lands
	on a source location
	"""
	# paths of audio files
	paths = choose_random_files()

	# dimensions of room
	room_config = [10, 10]

	# locations of audio sources (direct correspondence with paths list, e.g. [5,5] is the location of 050a050a.wav)
	#source_loc = [[0, 0], [9, 9]]  # Keep the sources far away for easy testing

	# location of agent (or microphone in this case)
	agent_loc = np.array([5, 5])

	# Set up the gym environment
	env = gym.make('audio-room-v0', room_config=room_config, agent_loc=agent_loc)
	env.add_sources(direct_sources=paths)

	# Load the agent class
	agent = rl_agent.RandomAgent()
	agent.fit(env)


def run_perfect_agent():
	"""
	This function runs the Perfect Agent in the ShoeBox audio environment. The agent randomly picks a target and then
	goes to the x value and then the y value until it arrives.
	"""
	# paths of audio files
	paths = choose_random_files()

	# dimensions of room
	room_config = [15, 15]

	# NOTE: now going to randomly place source in room
	# locations of audio sources (direct correspondence with paths list)
	#source_loc = [[2, 2], [14, 14]]  # Keep the sources far away for easy testing

	# location of agent (or microphone in this case)
	# agent_loc = np.array([11, 11])

	# Set up the gym environment
	env = gym.make('audio-room-v0', room_config=room_config, num_channels=2, bytes_per_sample=2)
	env.add_sources(direct_sources=paths)

	# ---- Only for debuggin ---
	# # env.room.plot()
	# # plt.show()
	# env.step(1, play_audio=False, show_room=False)
	# env.room.plot(img_order=0)
	# plt.show()
	# -------------------------
	# Load the agent class
	target_loc = env.target
	agent = rl_agent.PerfectAgent(
		target_loc=target_loc, agent_loc=env.agent_loc, play_audio=True, show_room=True)
	print("about to fit")
	agent.fit(env)


def run_room_agent_oroom1():
	"""
	This function runs the ORoom Agent in the ORoom audio environment. The point of this class
	is to create an environment different than the ShoeBox room.

	NOTE: this can no longer be run with random source placement
		- may not need this since staying with simpler rooms
	"""
	# paths of audio files
	paths = choose_random_files()

	# dimensions of room (Horseshoe room, see class for visualization); order matters!
	room_config = np.array(
		[[2, 2], [2, 10], [5, 10], [5, 5], [8, 5], [8, 10], [10, 10], [10, 2]]).T

	# locations of audio sources (direct correspondence with paths list)
	source_loc = [[6, 4.5], [9, 8]]

	# this agent will go get 9,8
	# agent_loc = np.array([3, 8])
	agent_loc = np.array([3, 8])

	# Set up the gym environment
	env = gym.make('audio-room-v0', room_config=room_config, agent_loc=agent_loc, corners=True, max_order=10)
	env.add_sources(direct_sources=paths, target=1)  # target is the 2nd source
	# env.room.plot()
	# plt.show()
	# env.step(3)
	# env.room.plot()
	# plt.show()
	# Load the agent class
	target_loc = env.target
	agent = rl_agent.PerfectAgentORoom(target_loc=target_loc, agent_loc=agent_loc)
	agent.fit(env)


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


if __name__ == '__main__':
	# Run the random agent
	#run_random_agent()
	# Run perfect agent
	run_perfect_agent()
	# run_room_agent_oroom1()
