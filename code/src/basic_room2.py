import gym
import audio_room
import numpy as np
import rl_agent
import os

DIR_MALE = '../sounds/dry_recordings/dev/051_subset/'
DIR_FEMALE = '../sounds/dry_recordings/dev/050_subset/'

def run_random_agent():
	# paths of audio files
	paths = choose_random_files()

	# dimensions of room
	room_config = [10, 10]

	# locations of audio sources (direct correspondence with paths list, e.g. [5,5] is the location of 050a050a.wav)
	source_loc = [[0, 0], [9, 9]]  # Keep the sources far away for easy testing

	# location of agent (or microphone in this case)
	agent_loc = np.array([5, 5])

	# Set up the gym environment
	env = gym.make('audio-room-v0', room_config=room_config, agent_loc=agent_loc)
	env.add_sources(direct_sound=paths, sound_loc=source_loc)

	# Load the agent class
	agent = rl_agent.RandomAgent()
	agent.fit(env)


def run_perfect_agent():
	# paths of audio files
	paths = choose_random_files()

	# dimensions of room
	room_config = [30, 30]

	# locations of audio sources (direct correspondence with paths list, e.g. [5,5] is the location of 050a050a.wav)
	source_loc = [[0, 0], [14, 14]]  # Keep the sources far away for easy testing

	# location of agent (or microphone in this case)
	agent_loc = np.array([11, 11])

	# Set up the gym environment
	env = gym.make('audio-room-v0', room_config=room_config, agent_loc=agent_loc)
	env.add_sources(direct_sound=paths, sound_loc=source_loc)

	# Load the agent class
	# target_loc = env.target
	target_loc = env.target
	agent = rl_agent.PerfectAgent(target_loc=target_loc, agent_loc=agent_loc)
	agent.fit(env)


def choose_random_files():
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
 