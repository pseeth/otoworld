import gym
import audio_room
import numpy as np
import rl_agent
import os

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
	"""
	This function runs the Perfect Agent in the ShoeBox audio environment. The agent randomly picks a target and then
	goes to the x value and then the y value until it arrives.
	"""
	# paths of audio files
	paths = choose_random_files()

	# dimensions of room
	room_config = [30, 30]

	# locations of audio sources (direct correspondence with paths list)
	source_loc = [[2, 2], [14, 14]]  # Keep the sources far away for easy testing

	# location of agent (or microphone in this case)
	agent_loc = np.array([11, 11])

	# Set up the gym environment
	env = gym.make('audio-room-v0', room_config=room_config, agent_loc=agent_loc)
	env.add_sources(direct_sound=paths, sound_loc=source_loc)

	# Load the agent class
	target_loc = env.target
	agent = rl_agent.PerfectAgent(target_loc=target_loc, agent_loc=agent_loc)
	agent.fit(env)


def run_perfect_horseshoe_room_agent():
	"""
	This function runs the Perfect Horseshoe Room Agent in the Horseshoe Room audio environment. The Horseshoe room
	is created by just passing the corners of the room as the room_config. For now, PyRoom seems unable to compute
	the RIR.
	"""
	# paths of audio files
	paths = choose_random_files()

	# dimensions of room (Horseshoe room, see class for visualization); order matters!
	room_config = np.array([[2, 2], [2, 10], [5, 10], [5, 5], [8, 5], [8, 10], [10, 10], [10, 2]]).T

	# locations of audio sources (direct correspondence with paths list)
	source_loc = [[6, 4.5], [9, 8]]

	# this agent will go get 9,8
	agent_loc = np.array([3, 8])

	# Set up the gym environment
	env = gym.make('audio-room-v0', room_config=room_config, agent_loc=agent_loc, corners=True)
	env.add_sources(direct_sound=paths, sound_loc=source_loc, target=1)  # target is the 2nd source

	# Load the agent class
	target_loc = env.target
	agent = rl_agent.PerfectAgentHorseshoeRoom(target_loc=target_loc, agent_loc=agent_loc)
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
	#run_perfect_agent()
	# Run the horseshoe agent
	run_perfect_horseshoe_room_agent()
 