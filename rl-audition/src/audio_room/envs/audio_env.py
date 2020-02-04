import gym
from pyroomacoustics import MicrophoneArray, ShoeBox, Room, linear_2D_array
import librosa
import numpy as np
import matplotlib.pyplot as plt
import simpleaudio as sa
from gym import spaces
from random import randint
import time


class AudioEnv(gym.Env):
	def __init__(self, room_config, agent_loc=None, resample_rate=8000, num_channels=2, bytes_per_sample=2, corners=False, absorption=0.0, max_order=2):
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
			self.room = Room.from_corners(room_config, fs=resample_rate, absorption=absorption, max_order=max_order)
			self.agent_loc = agent_loc
			print(room_config[0])
			print(room_config[1])

			# The x_max and y_max in this case would be used to generate agent's location randomly
			self.x_max = min(room_config[0])  # The minimum is important
			self.y_max = min(room_config[1])

		# NOTE: this rl-audition assumes ShoeBox config and that default arg
		else:
			self.room = ShoeBox(room_config, absorption=absorption)
			self.x_max = room_config[0]-1
			self.y_max = room_config[1]-1

		if agent_loc is not None:
			self.agent_loc = agent_loc
		else:
			# Generate initial agent location randomly if nothing is specified
			x = randint(0, self.x_max)
			y = randint(0, self.y_max)
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
			self.target = sound_loc[target]
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

		if self.num_channels == 2:
			# Create the array at current time step (2 mics, angle 0, 0.5m apart)
			mic = MicrophoneArray(linear_2D_array(agent_loc, 2, 15, 0.5), self.room.fs)
			self.room.add_microphone_array(mic)
		else:
			mic = MicrophoneArray(agent_loc.reshape(-1, 1), self.room.fs)
			self.room.add_microphone_array(mic)

	def check_if_inside(self, points):
		return self.room.is_inside(points, include_borders=False)

	def step(self, action, play_audio=True, show_room=True):
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
			show_room (bool): Controls whether room is visually plotted or not

		Returns:
			Tuple of the format List (empty if done, else [data]), reward, done
		"""
		x, y = self.agent_loc[0], self.agent_loc[1]
		done = False
		if action == 0:
			x -= 1
		elif action == 1:
			x += 1
		elif action == 2:
			y += 1
		elif action == 3:
			y -= 1

		# Check if the new points lie within the room
		points = np.array([x, y]) if self.room.is_inside([x, y], include_borders=False) else self.agent_loc
		print("Agent performed action: ", self.action_to_string[action])
		# Move agent in the direction of action
		self._move_agent(agent_loc=points)
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
			# print("Mic array", self.room.mic_array.R.T.shape)
			# print("Sources", len(self.room.sources))
			# for m, mic in enumerate(self.room.mic_array.R.T):
			# 	h = []
			# 	for s, source in enumerate(self.room.sources):
			# 		print("Source: ", source)
			# 		print("Visibility: ", self.room.is_visible(source, self.agent_loc))
			# 		print(self.room.visibility)
			# 		h.append(source.get_rir(mic, self.room.visibility[s][m], self.room.fs, self.room.t0))
			# 	self.room.rir.append(h)
			# print("Max order: ", self.room.max_order)
			self.room.compute_rir()
			self.room.simulate()
			data = self.room.mic_array.signals
			# print(data.shape)

			# # print RIR
			# self.room.plot_rir()
			# fig = plt.gcf()
			# fig.set_size_inches(16, 8)
			# plt.show()

			if play_audio or show_room:
				self.render(data, play_audio, show_room)

			# Calculate the reward
			# Currently, simply using negative of l2 norm as reward
			reward = -np.linalg.norm(self.agent_loc - self.target)

			# Return the room rir and convolved signals as the new state
			return [data], reward, done

	def reset(self):
		"""
		This function resets the agent to a random location within the room. To be used after each episode. NOTE: this
		rl-audition assumes ShoeBox config.
		"""
		# Reset agent's location
		# Generate initial agent location randomly if nothing is specified
		x = randint(0, self.x_max)
		y = randint(0, self.y_max)
		self.agent_loc = [x, y]

		print("Agent location reset to: ", self.agent_loc)

	def render(self, data, play_audio, show_room):
		"""
		Play the convolved sound using SimpleAudio.

		Args:
			data (np.array): if 2 mics, should be of shape (x, 2)
			play_audio (bool): If true, audio will play
			show_room (bool): If true, room will be displayed to user
		"""
		if play_audio:
			scaled = np.zeros((data.shape[1], data.shape[0]))

			# Scale each microphone separately -> Important
			scaled[:, 0] = data[0] / np.max(np.abs(data[0])) * 32767
			scaled[:, 1] = data[1] / np.max(np.abs(data[1])) * 32767
			# Int16 is required to play the audio correctly
			scaled = scaled.astype(np.int16)
			# print("Scaled", scaled.shape)pe)
			play_obj = sa.play_buffer(scaled, num_channels=self.num_channels, bytes_per_sample=self.bytes_per_sample,
									  sample_rate=self.resample_rate)

			# Show the room while the audio is playing
			if show_room:
				fig, ax = self.room.plot(img_order=0)
				plt.pause(0.001)

			play_obj.wait_done()
			plt.close()

