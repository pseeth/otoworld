import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pyroomacoustics as pra
from pyroomacoustics import MicrophoneArray, ShoeBox
import librosa
import numpy as np
import matplotlib.pyplot as plt
import simpleaudio as sa
from gym import spaces
from random import randint
import time


class AudioEnv(gym.Env):
	def __init__(self, room_config, agent_loc=None, resample_rate=8000, num_channels=1, bytes_per_sample=2, corners=False):
		"""
		This class inherits from OpenAI Gym Env and is used to simulate the agent moving in PyRoom.

		Args:
			room_config (List or np.array): dimensions of the room. For Shoebox, in the form of [10,10]. Otherwise,
				in the form of [[1,1], [1, 4], [4, 4], [4, 1]] specifying the corners of the room
			agent_loc (List or np.array): initial location of the agent (mic array).
			resample_rate (int): sample rate in Hz
			num_channels (int): number of channels (used in playing what the mic hears)
			bytes_per_sample (int): used in playing what the mic hears
			corners (bool): False if using Shoebox config, otherwise True
		"""
		self.resample_rate = resample_rate
		self.audio = []
		self.num_channels = num_channels
		self.bytes_per_sample = bytes_per_sample
		self.num_actions = 4  # For now, having 4 actions = left, right, up, down
		self.action_space = spaces.Discrete(self.num_actions)
		self.target = None
		self.action_to_string = {0: 'Left', 1: 'Right', 2: 'Up', 3: 'Down'}
		self.corners = corners

		if self.corners:
			self.room = pra.Room.from_corners(room_config, fs=resample_rate)
			self.agent_loc = agent_loc

			# these maxes don't really make sense but the rest of the code was written to need them
			self.x_max = max(room_config[0])
			self.y_max = max(room_config[1])

		# NOTE: this code assumes ShoeBox config and that default arg
		else:
			self.room = ShoeBox(room_config)
			self.x_max = room_config[0]
			self.y_max = room_config[1]
			if agent_loc is not None:
				self.agent_loc = agent_loc
			else:
				# Generate initial agent location randomly if nothing is specified
				x = randint(0, self.x_max-1)
				y = randint(0, self.y_max-1)
				self.agent_loc = [x, y]

		print("Initial agent location: ", self.agent_loc)

	def add_sources(self, direct_sound, sound_loc, target=None):
		"""
		This function adds the sources to PyRoom

		Args:
			direct_sound (str): The path to sound files
			sound_loc (List[int]): A list consisting of [x, y] coordinates of source location
			target (int): The index value of sound_loc list which is to be set as target
		"""
		for idx, audio_file in enumerate(direct_sound):
			# Audio will be automatically re-sampled to the given rate (default sr=8000).
			a, _ = librosa.load(audio_file, sr=self.resample_rate)

			# If sound is recorded on stereo microphone, it is 2d
			# Take the mean of the two stereos
			if len(a.shape) > 1:
				a = np.mean(a, axis=0)

			self.audio.append(a)
			self.room.add_source(sound_loc[idx], signal=a)

		# if not Shoebox config
		if self.corners:
			self.target = target
		else:
			# One of the sources would be the target source; i.e the one which agent will move to
			if target is not None:
				self.target = sound_loc[target]
			else:
				# Set a random target otherwise
				self.target = sound_loc[randint(0, len(sound_loc)-1)]

		print("The target source is set as: ", self.target)
		self.target = np.array(self.target)

	def _move_agent(self, agent_loc):
		"""
		This function moves the agent to a new location (given by agent_loc). It effectively removes the
		agent (mic array) from the room and then adds it back in the new location.

		Args:
			agent_loc (List[int] or np.array): [x,y] coordinates of the agent's new location
		"""
		# Set the new agent location
		self.agent_loc = agent_loc

		# Delete the array at previous time step
		self.room.mic_array = None

		# Create the array at current time step (2 mics, angle 0, 2m apart)
		mic = pra.linear_2D_array(agent_loc, 2, 0, .5)
		self.room.add_microphone_array(MicrophoneArray(mic, self.room.fs))

		# Plot the room
		self.room.plot()
		plt.show()

	def step(self, action, play_audio=True):
		"""
		This function simulates the agent taking one step in the environment (and room) given an action:
			0 = Left
			1 = Right
			2 = Up
			3 = Down
		It calls _move_agent, checks to see if the agent has reached a target, and if not, computes the RIR.

		Args:
			action (int): direction agent is to move - 0 (L), 1 (R), 2 (U), 3 (D)
			play_audio (bool): whether to play the the mic audio (stored in "data")

		Returns:
			Tuple of the format List (empty if done, else [data]), reward, done
		"""
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
			data = self.room.mic_array.signals

			# print RIR
			self.room.plot_rir()
			fig = plt.gcf()
			fig.set_size_inches(16, 8)
			plt.show()

			if play_audio:
				self.render(data)

			# Calculate the reward
			# Currently, simply using negative of l2 norm as reward
			reward = -np.linalg.norm(self.agent_loc - self.target)

			# Return the room rir and convolved signals as the new state
			return [data], reward, done

	def reset(self):
		"""
		This function resets the agent to a random location within the room. To be used after each episode. NOTE: this
		code assumes ShoeBox config.
		"""
		# Reset agent's location
		# Generate initial agent location randomly if nothing is specified
		x = randint(0, self.x_max - 1)
		y = randint(0, self.y_max - 1)
		self.agent_loc = [x, y]

		print("Agent location reset to: ", self.agent_loc)

	def render(self, data):
		"""
		Play the convolved sound using SimpleAudio.

		Args:
			data (np.array): if 2 mics, should be of shape (2, x)
		"""
		# Scaling of data is important, otherwise audio won't play
		#data = data.reshape(-1, 2)
		scaled = np.int16(data / np.max(np.abs(data)) * 32767)
		play_obj = sa.play_buffer(scaled, num_channels=self.num_channels, bytes_per_sample=self.bytes_per_sample,
								  sample_rate=self.resample_rate)
		play_obj.wait_done()

