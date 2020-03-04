import numpy as np
import gym
import warnings
import time
from scipy.spatial.distance import euclidean
import store_data
from collections import deque


class RLAgent:
    def __init__(self, episodes=1, steps=10, blen=1000, gamma=0.9, alpha=0.001):
        """
		This class is a wrapper for the actual RL agent

		Args:
			episodes (int): # of episodes to simulate
			steps (int): # of steps the agent can take before stopping an episode
			blen (int): # of entries which the replay buffer can store
			gamma (float): Discount factor
			alpha (float): Learning rate alpha
		"""
        self.episodes = episodes
        self.max_steps = steps
        self.blen = blen
        self.buffer = deque(maxlen=blen)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = 1.0  # Exploration rate
        self.max_epsilon = 1.0  # Max value of epsilon and also the starting exploration rate
        self.min_epsilon = 0.01
        self.decay_rate = 0.005  # Decay rate for exploration rate (We want to decrease exploration as time proceeds)
        self.play_audio = False
        self.show_room = False

    def fit(self, env):
        for episode in range(self.episodes):
            # Reset the environment and any other variables at beginning of each episode
            env.reset()
            prev_state = None
            done = False
            print("Value of epsilon: ", self.epsilon)

            for step in range(self.max_steps):

                # Perform random actions with prob < epsilon
                random_prob = np.random.uniform(0, 1)
                if random_prob < self.epsilon:
                    action = env.action_space.sample()
                else:
                    """
					Agent will decide the action here. Call the agent here 
					"""
                    # If it is the first step (prev_state is zero), then perform a random action
                    if step == 0:
                        action = env.action_space.sample()
                    else:
                        # This is where agent will actually do something
                        pass

                # Perform the chosen action
                new_state, target, reward, done = env.step(
                    action, play_audio=self.play_audio, show_room=self.show_room
                )

                # Perform the q-update or whatever we are using over here
                """
				Update q network 
				"""

                if step > 0:
                    self.buffer.append((prev_state, action, new_state, reward))

                prev_state = new_state

                # Terminate the episode if done
                if done:
                    break

            # Decay the epsilon
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
                -self.decay_rate * episode
            )
