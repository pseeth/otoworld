import gym
from pyroomacoustics import MicrophoneArray, ShoeBox, Room, linear_2D_array, Constants
import librosa
import numpy as np
import matplotlib.pyplot as plt
#import simpleaudio as sa  # comment out for gpubox
from gym import spaces
from random import randint
import time
from scipy.spatial.distance import euclidean
from copy import deepcopy
import nussl
import sys

sys.path.append("../../")
import constants


class AudioEnv(gym.Env):
    def __init__(
        self,
        room_config,
        agent_loc=None,
        resample_rate=8000,
        num_channels=2,
        bytes_per_sample=2,
        corners=False,
        absorption=0.0,
        max_order=2,
        converge_steps=10,
        step_size=1,
        acceptable_radius=0.1,
        direct_sources=None,
        degrees=0.2618,
    ):
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
            degrees (float): value of degrees to rotate in radians (.2618 radians = 15 degrees)
        """
        self.resample_rate = resample_rate
        self.absorption = absorption
        self.max_order = max_order
        self.audio = []
        self.num_channels = num_channels
        self.bytes_per_sample = bytes_per_sample
        self.num_actions = 6
        self.action_space = spaces.Discrete(self.num_actions)
        self.action_to_string = {
            0: "Left",
            1: "Right",
            2: "Up",
            3: "Down",
            4: "Rotate Left",
            5: "Rotate right",
        }
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
        self.min_size_audio = np.inf
        self.degrees = degrees
        self.cur_angle = 0  # The starting angle is 0

        # non-Shoebox config (corners of room are given)
        if self.corners:
            self.room = Room.from_corners(
                room_config, fs=resample_rate, absorption=absorption, max_order=max_order,
            )

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
        """
        This method would generate random sample points using rejection sampling method

        Args:
            num_sources: Number of (x, y) random points generated will be equal to number of sources

        Returns:
            A list of generated random points
        """

        sampled_points = []
        generated_points = {}  # To avoid placing multiple sources in the same location

        while len(sampled_points) < num_sources:
            random_point = [
                np.random.randint(self.x_min, self.x_max),
                np.random.randint(self.y_min, self.y_max),
            ]
            try:
                if (
                    self.room.is_inside(random_point, include_borders=False)
                    and tuple(random_point) not in generated_points
                ):
                    sampled_points.append(random_point)
                    generated_points[tuple(random_point)] = 1
            except:
                # in case is_inside func fails, place in the center for now
                point = [
                    ((self.x_max - self.x_min) // 2),
                    ((self.y_max - self.y_min) // 2),
                ]
                sampled_points.append(point)
                generated_points[tuple(point)] = 1

        return sampled_points

    def add_sources(self, source_locs=None, reset_env=False, removing_source=None):
        """
        This function adds the sources to PyRoom. Assumes 2 sources.

        Args:
            source_loc (List[int]): A list consisting of [x, y] coordinates of source location
            reset_env (bool): Bool indicating whether we reset_env the agents position to be the mean
                of all the sources
            removing_source (None or int): Value that will tell us if we are removing a source
                from sources
        """
        # If we are reset_env-ing our env, we have to get the original sources
        if reset_env:
            self.direct_sources = deepcopy(self.direct_sources_copy)
        # If we are removing a source, we remove from direct sources and source locs
        elif removing_source is not None:
            self.source_locs.pop(removing_source)
            self.direct_sources.pop(removing_source)

        # Place sources in room if we need to reset_env or if sources are none
        if self.source_locs is None or reset_env:
            if source_locs is None:
                # Generate random points using rejection sampling method
                self.source_locs = self._sample_points(num_sources=len(self.direct_sources))
            else:
                self.source_locs = source_locs

        # Resetting the agents position to be the mean of all sources
        if self.agent_loc is None or reset_env:
            self.agent_loc = np.mean(self.source_locs, axis=0)

        self.audio = []
        self.min_size_audio = np.inf
        for idx, audio_file in enumerate(self.direct_sources):
            # Audio will be automatically re-sampled to the given rate (default sr=8000).
            a = nussl.AudioSignal(audio_file, sample_rate=self.resample_rate)
            # If sound is recorded on stereo microphone, it is 2d
            if a.is_stereo:
                a.to_mono()

            # normalize audio so both sources have similar volume at beginning before mixed
            a.peak_normalize()

            # Finding the minimum size source to make sure there is something playing at all times
            if len(a) < self.min_size_audio:
                self.min_size_audio = len(a)
            self.audio.append(a.audio_data.squeeze())

        # add sources using audio data
        for idx, audio in enumerate(self.audio):
            self.room.add_source(self.source_locs[idx], signal=audio[: self.min_size_audio])

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
            mic = MicrophoneArray(linear_2D_array(agent_loc, 2, self.cur_angle, 0.2), self.room.fs)
            self.room.add_microphone_array(mic)
        else:
            mic = MicrophoneArray(agent_loc.reshape(-1, 1), self.room.fs)
            self.room.add_microphone_array(mic)

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

        It calls _move_agent, checks to see if the agent has reached a source, and if not, computes the RIR.

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
        try:
            if self.room.is_inside([x, y], include_borders=False):
                points = np.array([x, y])
            else:
                points = self.agent_loc
        except:
            # in case the is_inside func fails
            points = self.agent_loc

        # Move agent in the direction of action
        self._move_agent(agent_loc=points)

        # Check if goal state is reached
        """
        If agent loc exactly matches target location then pyroomacoustics isn't able to 
        calculate the convolved signal. Hence, check the location before calculating everything   
        """
        for index, source in enumerate(self.source_locs):
            # Agent has reach the goal if the agent is with the circle around the source
            if euclidean(self.agent_loc, source) < self.acceptable_radius:
                # If there is more than one source, then we want to remove this source
                if len(self.source_locs) > 1:
                    # remove the current source and reset_env the environment
                    self.reset(removing_source=index)

                    # Calculate the impulse response
                    self.room.compute_rir()
                    self.room.simulate()
                    data = self.room.mic_array.signals
                    # Convert the data back to Nussl Audio object
                    data = nussl.AudioSignal(audio_data_array=data, sample_rate=self.resample_rate)

                    if play_audio or show_room:
                        self.render(data, play_audio, show_room)

                    done = False
                    reward = constants.TURN_OFF_REWARD
                    # Return the room rir and convolved signals as the new state
                    return data, reward, done

                # This was the last source hence we can assume we are done
                else:
                    done = True
                    reward = constants.TURN_OFF_REWARD
                    return None, reward, done

        if not done:
            # Calculate the impulse response
            self.room.compute_rir()
            self.room.simulate()
            data = self.room.mic_array.signals
            # Convert data to nussl audio signal
            data = nussl.AudioSignal(audio_data_array=data, sample_rate=self.resample_rate)

            if play_audio or show_room:
                self.render(data, play_audio, show_room)

            # penalize time it takes to reach a source
            reward = constants.STEP_PENALTY

            # Return the room rir and convolved signals as the new state
            return data, reward, done

    def reset(self, removing_source=None):
        """
        This function reset_envs the sources to a random location within the room. To be used after each episode.

        Currently: the agent is placed back in initial location (make random eventually)

        args:
            removing_source (int): Integer that tells us the index of sources that we will be removing
        """
        # Generate initial agent location randomly in the future
        # non-Shoebox config (corners of room are given)
        if self.corners:
            self.room = Room.from_corners(
                self.room_config,
                fs=self.resample_rate,
                absorption=self.absorption,
                max_order=self.max_order,
            )

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
            self.add_sources(reset_env=True)

    def render(self, data, play_audio, show_room):
        """
        Play the convolved sound using SimpleAudio.

        Args:
            data (AudioSignal): if 2 mics, should be of shape (x, 2)
            play_audio (bool): If true, audio will play
            show_room (bool): If true, room will be displayed to user
        """
        if play_audio:
            data.play()

            # Show the room while the audio is playing
            if show_room:
                fig, ax = self.room.plot(img_order=0)
                plt.pause(1)

            plt.close()

        elif show_room:
            fig, ax = self.room.plot(img_order=0)
            plt.pause(1)
            plt.close()
