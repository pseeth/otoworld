import sys
sys.path.append('../src/')
import gym
import audio_room
import numpy as np
from agent import RandomAgent
import os
import matplotlib.pyplot as plt
import room_types
import utils

DIR_MALE = '../sounds/dry_recordings/dev/051_subset/'
DIR_FEMALE = '../sounds/dry_recordings/dev/050_subset/'


def run_random_agent():
	"""
	This function runs the Random Agent in the ShoeBox audio environment. The agent randomly moves around until it lands
	on a source location
	"""
	# paths of audio files
	paths = utils.choose_random_files()

	# dimensions of room
	room_config = [10, 10]

	# locations of audio sources (direct correspondence with paths list, e.g. [5,5] is the location of 050a050a.wav)
	#source_loc = [[0, 0], [9, 9]]  # Keep the sources far away for easy testing

	# location of agent (or microphone in this case)
	agent_loc = np.array([5, 5])

	# Set up the gym environment
	env = gym.make('audio-room-v0', room_config=room_config, agent_loc=agent_loc, direct_sources=paths)
	env.add_sources()

	# Load the agent class
	agent = RandomAgent()
	agent.fit(env, show_room=False, play_audio=False)


# run_random_agent()

if __name__ == '__main__':
	run_random_agent()
	# import nussl
	# paths = utils.choose_random_files()
	# print(paths)
	#
	#
	#
	# import numpy as np
	# import matplotlib.pyplot as plt
	# import pyroomacoustics as pra
	# import librosa
	#
	# # Create a 4 by 6 metres shoe box room
	# room = pra.ShoeBox([10, 10])
	#
	# # Add a source somewhere in the room
	# # room.add_source([2.5, 4.5])
	#
	# # Create a linear array beamformer with 4 microphones
	# # with angle 0 degrees and inter mic distance 10 cm
	# # R = pra.linear_2D_array([2, 1.5], 4, 0, 0.04)
	# # room.add_microphone_array(pra.Beamformer(R, room.fs))
	# mic = pra.MicrophoneArray(np.array([2, 1.5]).reshape(-1, 1), room.fs)
	# room.add_microphone_array(mic)
	# # Now compute the delay and sum weights for the beamformer
	# pos = [[1.5, 3.5], [5.5, 8.5]]
	# for i, source in enumerate(paths):
	# 	s = nussl.AudioSignal(source, sample_rate=8000)
	# 	l, _ = librosa.load(source, sr=8000)
	# 	if s.is_stereo:
	# 		s = s.to_mono()
	#
	# 	print(len(s), s.audio_data.shape, l.shape, s.num_channels, _)
	# 	room.add_source(pos[i], s.audio_data.squeeze())
	# 	print(s.is_mono, s.is_stereo)
	#
	# room.compute_rir()
	# room.simulate()
	# data = room.mic_array.signals
	#
	# # plot the room and resulting beamformer
	# room.plot(img_order=0)
	# plt.show()