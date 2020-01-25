import gym
from gym import error, spaces, utils
from gym.utils import seeding
from pyroomacoustics import MicrophoneArray, ShoeBox
import librosa
import numpy as np
import matplotlib.pyplot as plt
import simpleaudio as sa
from gym import spaces
from random import randint


class AudioEnv(gym.Env):
	def __init__(self, room_config, agent_loc=None, resample_rate=8000, num_channels=1, bytes_per_sample=2):
		self.resample_rate = resample_rate
		self.room = ShoeBox(room_config)
		self.x_max = room_config[0]
		self.y_max = room_config[1]
		self.audio = []
		self.num_channels = num_channels
		self.bytes_per_sample = bytes_per_sample
		self.num_actions = 4  # For now, having 4 actions = left, right, up, down
		self.action_space = spaces.Discrete(self.num_actions)
		self.target = None
		self.action_to_string = {0: 'Left', 1: 'Right', 2: 'Up', 3: 'Down'}

		if agent_loc is not None:
			self.agent_loc = agent_loc
		else:
			# Generate initial agent location randomly if nothing is specified
			x = randint(0, self.x_max-1)
			y = randint(0, self.y_max-1)
			self.agent_loc = [x, y]

		print("Initial agent location: ", self.agent_loc)

	def add_sources(self, direct_sound, sound_loc, target=None):
		'''
		:param direct_sound: The path to sound files
		:param sound_loc: A list consisting of [x, y] co-ordinates of source location
		:param target: The index value of sound_loc list which is to be set as target
		:return:
		'''
		for idx, audio_file in enumerate(direct_sound):
			# Audio will be automatically resampled to the given rate (default sr=8000).
			a, _ = librosa.load(audio_file, sr=self.resample_rate)

			# If sound is recorded on stereo microphone, it is 2d
			# Take the mean of the two stereos
			if len(a.shape) > 1:
				a = np.mean(a, axis=0)

			self.audio.append(a)
			self.room.add_source(sound_loc[idx], signal=a)

		# One of the sources would be the target source; i.e the one which agent will move to
		if target is not None:
			self.target = sound_loc[target]
		else:
			# Set a random target otherwise
			self.target = sound_loc[randint(0, len(sound_loc)-1)]

		print("The target source is set as: ", self.target)
		self.target = np.array(self.target)

	def _move_agent(self, agent_loc):
		# Set the new agent location
		self.agent_loc = agent_loc
		# Delete the array at previous time step
		self.room.mic_array = None
		# Create the array at current time step
		mic = MicrophoneArray(agent_loc.reshape(-1, 1), self.room.fs)
		self.room.add_microphone_array(mic)

	def step(self, action, play_audio=True):
		'''
		0 = Left
		1 = Right
		2 = Up
		3 = Down
		'''
		x, y = self.agent_loc[0], self.agent_loc[1]
		done = False
		if action == 0:
			if x != 0:
				x -= 1
		elif action == 1:
			if x != self.x_max-1:
				x += 1
		elif action == 2:
			if y != self.y_max-1:
				y += 1
		elif action == 3:
			if y != 0:
				y -= 1
		# Move agent in the direction of action
		print("Agent performed action: ", self.action_to_string[action])
		self._move_agent(agent_loc=np.array([x, y]))
		print("New agent location: ", self.agent_loc)
		print("Target location: ", self.target)
		print("---------------------")
		# Check if goal state is reached
		'''
		If agent loc exactly matches target location then pyroomacoustics isn't able to 
		calculate the convolved signal. Hence, check the location before calculating everything   
		'''
		if self.agent_loc[0] == self.target[0] and self.agent_loc[1] == self.target[1]:
			print("Goal state reached!")
			done = True
			reward = 100

			return [], reward, done

		if not done:
			# Calculate the impulse response
			self.room.compute_rir()
			self.room.simulate()
			data = self.room.mic_array.signals[0, :]

			if play_audio:
				self.render(data)

			# Calculate the reward
			# Currently, simply using negative of l2 norm as reward
			reward = -np.linalg.norm(self.agent_loc - self.target)

			# Return the room rir and convolved signals as the new state
			return [self.room.rir, data], reward, done

	def reset(self):
		# Reset agent's location
		# Generate initial agent location randomly if nothing is specified
		x = randint(0, self.x_max - 1)
		y = randint(0, self.y_max - 1)
		self.agent_loc = [x, y]

		print("Agent location reset to: ", self.agent_loc)

	def render(self, data):
		# Play the audio using Simple Audio
		# Scaling of data is important, otherwise audio won't play
		scaled = np.int16(data / np.max(np.abs(data)) * 32767)
		play_obj = sa.play_buffer(scaled, num_channels=self.num_channels, bytes_per_sample=self.bytes_per_sample,
								  sample_rate=self.resample_rate)
		play_obj.wait_done()

