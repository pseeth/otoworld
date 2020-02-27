import gym
from pyroomacoustics import MicrophoneArray, ShoeBox, Room, linear_2D_array
import librosa
import numpy as np
import matplotlib.pyplot as plt
import simpleaudio as sa
from gym import spaces
from random import randint
import time
from scipy.spatial.distance import euclidean
from copy import deepcopy


class AudioEnv(gym.Env):
	def __init__(self, room_config, agent_loc=None, resample_rate=8000, num_channels=2, bytes_per_sample=2, corners=False,
              absorption=0.0, max_order=2, converge_steps=10, step_size=None,
				 acceptable_radius=.1, direct_sources=None, target=None, degrees=0.2618):
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
			absorption (float): Absorption param of the room (how walls absorb sound)
			max_order (int): another room parameter
			converge_steps (int): # of steps the perfect agent should make before rewards
			step_size (float): specificed step size else we programmatically assign it
			acceptable_radius (float): radius of acceptable range the agent can be in to be considered done
			direct_sources (List[str]): list of path strings to the source audio files
			target (int): index of which source in direct_sources is to be set as the target source (remove later)
		"""
		self.resample_rate = resample_rate
		self.absorption = absorption
		self.max_order = max_order
		self.audio = []
		self.num_channels = num_channels
		self.bytes_per_sample = bytes_per_sample
		self.num_actions = 6  # For now, having 6 actions = left, right, up, down, rotate left, rotate right
		self.action_space = spaces.Discrete(self.num_actions)
		self.action_to_string = {0: 'Left', 1: 'Right', 2: 'Up', 3: 'Down', 4: 'Rotate Left', 5: 'Rotate right'}
		self.corners = corners
		self.room_config = room_config
		self.agent_loc = agent_loc
		self.initial_agent_loc = agent_loc
		self.acceptable_radius = acceptable_radius
		self.converge_steps = converge_steps
		self.step_size = step_size
		self.direct_sources = direct_sources
		self.direct_sources_copy = deepcopy(direct_sources)
		self.source_locs = None
		self.target_source = None
		self.target = target
		self.min_size_audio = np.inf
		self.degrees = degrees
		self.cur_angle = 0  # The starting angle is 0

		# non-Shoebox config (corners of room are given)
		if self.corners:
			self.room = Room.from_corners(
				room_config, fs=resample_rate, absorption=absorption, max_order=max_order)

			# The x_max and y_max in this case would be used to generate agent's location randomly
			self.x_min = min(room_config[0])
			self.y_min = min(room_config[1])
			self.x_max = max(room_config[0])
			self.y_max = max(room_config[1])

		# ShoeBox config
		else:
			self.room = ShoeBox(room_config, absorption=absorption)
			self.x_max = room_config[0]
			self.y_max = room_config[1]
			self.x_min, self.y_min = 0, 0

		print("Initial agent location: ", self.agent_loc)

	def _sample_points(self, num_sources):
		'''
		This method would generate random sample points using rejection sampling method
		Args:
			num_sources: Number of (x, y) random points generated will be equal to number of sources

		Returns:
			A list of generated random points
		'''

		sampled_points = []
		generated_points = {}  # To avoid placing multiple sources in the same location

		while len(sampled_points) < num_sources:
			random_point = [np.random.randint(
				self.x_min, self.x_max), np.random.randint(self.y_min, self.y_max)]
			if self.room.is_inside(random_point, include_borders=False) and tuple(random_point) not in generated_points:
				sampled_points.append(random_point)
				generated_points[tuple(random_point)] = 1

		return sampled_points

	def add_sources(self, source_locs=None, reset=False, removing_source=None):
		"""
		This function adds the sources to PyRoom. Assumes 2 sources.

		Args:
			source_loc (List[int]): A list consisting of [x, y] coordinates of source location
			reset (bool): Bool indicating whether we reset the agents position to be the mean
				of all the sources
			removing_source (None or int): Value that will tell us if we are removing a source 
				from sources
		"""
		# If we are resetting our env, we have to get the original sources
		if reset:
			self.direct_sources = deepcopy(self.direct_sources_copy)
		# If we are removing a source, we remove from direct sources and source locs
		elif removing_source is not None:
			self.source_locs.pop(removing_source)
			self.direct_sources.pop(removing_source)

		# Place sources in room if we need to reset or if sources are none
		if self.source_locs is None or reset:
			if source_locs is None:
				# Generate random points using rejection sampling method
				self.source_locs = self._sample_points(
					num_sources=len(self.direct_sources))
			else:
				self.source_locs = source_locs

		# Resetting the agents position to be the mean of all sources
		if self.agent_loc is None or reset:
			self.agent_loc = np.mean(self.source_locs, axis=0)

		self.audio = []
		self.min_size_audio = np.inf
		for idx, audio_file in enumerate(self.direct_sources):
			# Audio will be automatically re-sampled to the given rate (default sr=8000).
			a, _ = librosa.load(audio_file, sr=self.resample_rate)

			# If sound is recorded on stereo microphone, it is 2d
			# Take the mean of the two stereos
			if len(a.shape) > 1:
				a = np.mean(a, axis=0)

			# normalize audio so both sources have similar volume at beginning before mixed
			a /= np.abs(a).max()

			# Finding the minimum size source to make sure there is something playing at all times
			if len(a) < self.min_size_audio:
				self.min_size_audio = len(a)
			self.audio.append(a)

		# add sources using audio data
		for idx, audio in enumerate(self.audio):
			self.room.add_source(
				self.source_locs[idx], signal=audio[:self.min_size_audio])

		# If we are removing a source, we have to choose a new target
		if removing_source is not None:
			self.target = randint(0, len(self.source_locs)-1)

		# if not Shoebox config
		if self.corners:
			self.target_source = self.source_locs[self.target]
		else:
			# One of the sources would be the target source; i.e the one which agent will move to
			if self.target is not None:
				self.target_source = self.source_locs[self.target]
			else:
				# Set a random target otherwise
				self.target = randint(0, len(self.source_locs)-1)
				self.target_source = self.source_locs[self.target]

		self.target_source = np.array(self.target_source)
		print("The target source is set as: ", self.target_source)
		print("Dist between src and agent:", euclidean(
			self.agent_loc, self.target_source))

		# Setting step size
		x_dis = abs(self.agent_loc[0] - self.target_source[0])
		y_dis = abs(self.agent_loc[1] - self.target_source[1])
		total_dis = x_dis + y_dis
		if self.step_size is None:
			self.step_size = (total_dis) / self.converge_steps

	def _move_agent(self, agent_loc):
		"""
		This function moves the agent to a new location (given by agent_loc). It effectively removes the
		agent (mic array) from the room and then adds it back in the new location.

		Args:
			agent_loc (List[int] or np.array): [x,y] coordinates of the agent's new location
			angle (int): discrete representation of angle to turn 
		"""
		# Set the new agent location
		self.agent_loc = agent_loc

		# Delete the array at previous time step
		self.room.mic_array = None

		if self.num_channels == 2:
			# Create the array at current time step (2 mics, angle IN RADIANS, 0.2m apart)
			mic = MicrophoneArray(linear_2D_array(
				agent_loc, 2, self.cur_angle, 0.2), self.room.fs)
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

		Agent can also simultaneously orient itself 15 degrees left or 15 degrees right
			0 = Don't orient
			1 = Orient 15 degrees left
			2 = Orient 15 degrees right

		It calls _move_agent, checks to see if the agent has reached a target, and if not, computes the RIR.

		Args:
			action (int): direction agent is to move - 0 (L), 1 (R), 2 (U), 3 (D)
			play_audio (bool): whether to play the the mic audio (stored in "data")
			show_room (bool): Controls whether room is visually plotted or not

		Returns:
			Tuple of the format List (empty if done, else [data]), reward, done
		"""
		x, y = self.agent_loc[0], self.agent_loc[1]
		# Separate out the action and orientation
		# action, angle = actions[0], actions[1]
		done = False
		if action == 0:
			x -= self.step_size
		elif action == 1:
			x += self.step_size
		elif action == 2:
			y += self.step_size
		elif action == 3:
			y -= self.step_size
		elif action == 4:
			self.cur_angle += self.degrees
		elif action == 5:
			self.cur_angle -= self.degrees
		# Check if the new points lie within the room
		points = np.array([x, y]) if self.room.is_inside(
			[x, y], include_borders=False) else self.agent_loc


		# print("Action taken: ", self.action_to_string[action],  " Cur angle: ", self.cur_angle, " In degrees: ", self.cur_angle*180/np.pi)
		# Move agent in the direction of action
		self._move_agent(agent_loc=points)
		dist = euclidean(self.agent_loc, self.target_source)

		# Check if goal state is reached
		'''
		If agent loc exactly matches target location then pyroomacoustics isn't able to 
		calculate the convolved signal. Hence, check the location before calculating everything   
		'''

		# Agent has reach the goal if the agent is with the circle around the target
		if euclidean(self.agent_loc, self.target_source) < self.acceptable_radius:
			print("Got a source!\n")
			# If there is more than one source, then we want to remove this source
			if len(self.source_locs) > 1:
				# remove the current source and reset the environment
				self.reset(removing_source=self.target)

				# Calculate the impulse response
				self.room.compute_rir()
				self.room.simulate()
				data = self.room.mic_array.signals

				if play_audio or show_room:
					self.render(data, play_audio, show_room)

				done = False
				reward = 100
				# Return the room rir and convolved signals as the new state
				return [data], self.target_source, reward, done

			# This was the last source hence we can assume we are done
			else:
				done = True
				reward = 100
				return [], None, reward, done

		if not done:
			# Calculate the impulse response
			self.room.compute_rir()
			self.room.simulate()
			data = self.room.mic_array.signals

			if play_audio or show_room:
				self.render(data, play_audio, show_room)

			# Calculate the reward
			# Currently, simply using negative of l2 norm as reward
			reward = -1 * np.linalg.norm(self.agent_loc - self.target_source)

			# Return the room rir and convolved signals as the new state
			return [data], self.target_source, reward, done

	def reset(self, removing_source=None):
		"""
		This function resets the sources to a random location within the room. To be used after each episode. 

		Currently: the agent is placed back in initial location (make random eventually)

		args: 
			removing_source (int): Integer that tells us the index of sources that we will be removing
		"""
		# Generate initial agent location randomly in the future
		# non-Shoebox config (corners of room are given)
		if self.corners:
			self.room = Room.from_corners(
				self.room_config, fs=self.resample_rate, absorption=self.absorption, max_order=self.max_order)

			# The x_max and y_max in this case would be used to generate agent's location randomly
			self.x_min = min(self.room_config[0])
			self.y_min = min(self.room_config[1])
			self.x_max = max(self.room_config[0])
			self.y_max = max(self.room_config[1])

		# ShoeBox config
		else:
			self.room = ShoeBox(self.room_config, absorption=self.absorption)
			self.x_max = self.room_config[0]
			self.y_max = self.room_config[1]
			self.x_min, self.y_min = 0, 0

		# Reset agent's location
		if removing_source is None:
			new_initial_agent_loc = self.initial_agent_loc
			self._move_agent(agent_loc=new_initial_agent_loc)
		else:
			self._move_agent(agent_loc=self.agent_loc)

		# We just remove the source
		if removing_source is not None:
			self.add_sources(removing_source=removing_source)
		# else add randomly generated new source locations if we are not removing a source
		else:
			self.add_sources(reset=True)

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
			scaled[:, 0] = data[0] * 32767
			scaled[:, 1] = data[1] * 32767
			# Int16 is required to play the audio correctly
			scaled = scaled.astype(np.int16)
			# print("Scaled", scaled.shape)pe)
			play_obj = sa.play_buffer(scaled, num_channels=self.num_channels, bytes_per_sample=self.bytes_per_sample,
                             sample_rate=self.resample_rate)

			# Show the room while the audio is playing
			if show_room:
				fig, ax = self.room.plot(img_order=0)
				plt.pause(1)

			play_obj.wait_done()
			plt.close()
