import time
import logging
import warnings

import numpy as np
import gym
from scipy.spatial.distance import euclidean
import nussl

import utils
import constants
from datasets import BufferData


class AgentBase:
    def __init__(
        self,
        env,
        dataset,
        episodes=1,
        steps=10,
        gamma=0.9,
        alpha=0.001,
        epsilon=1.0,
        decay_rate=0.005,
        play_audio=False,
        show_room=False,
        track_dist_vs_steps=False,
        plot_reward_vs_steps=False,
    ):
        """
        This class is a base agent class which will be inherited when creating various agents.

        Args:
            self.env (gym object): The gym self.environment object which the agent is going to explore
            dataset (nussl dataset): Nussl dataset object for experience replay
            episodes (int): # of episodes to simulate
            steps (int): # of steps the agent can take before stopping an episode
            gamma (float): Discount factor
            alpha (float): Learning rate alpha
            epsilon (float): Exploration rate, P(taking random action)
            decay_rate (float): decay rate for exploration rate (we want to decrease exploration as time proceeds)
            play_audio (bool): choose to play audio at each iteration
            show_room (bool): choose to display the configurations and movements within a room
            track_dist_vs_steps (bool): choose to track dist vs. num steps for each episode, use utils to log and plot
            plot_reward_vs_steps (bool): choose to track reward vs. num steps for each episode, use utils to log and plot
        """
        self.env = env
        self.dataset = dataset
        self.episodes = episodes
        self.max_steps = steps
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.play_audio = play_audio
        self.show_room = show_room
        self.track_dist_vs_steps = track_dist_vs_steps
        self.plot_reward_vs_steps = plot_reward_vs_steps

        # move this somewhere else?
        logging.basicConfig(
            filename='agent.log', 
            level=logging.INFO, 
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    
    def fit(self):
        if self.track_dist_vs_steps:
            init_dist_to_target = []
            steps_to_completion = []
        
        if self.plot_reward_vs_steps:
            rewards_per_episode = []

        logging_str = ('\n\n\nSTARTING FIT \n\n')
        logging.info(logging_str)

        for episode in range(self.episodes):
            # Reset the self.environment and any other variables at beginning of each episode
            self.env.reset()
            prev_state = None

            # Measure distance with the all sources
            if self.track_dist_vs_steps:
                init_dist_to_target.append(
                    sum([euclidean(self.env.agent_loc, source) for source in self.env.source_locs]))
                steps_to_completion.append(0)
            
            # Keep track of rewards for this episode
            if self.plot_reward_vs_steps:
                temp_rewards = []
            
            # Measure time to complete the episode
            start = time.time()
            for step in range(self.max_steps):
                if self.track_dist_vs_steps:
                    steps_to_completion[-1] += 1

                # Perform random actions with prob < epsilon
                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    # If it is the first step (prev_state is zero), then perform a random action
                    if step == 0:
                        action = self.env.action_space.sample()
                    else:
                        # This is where agent will actually do something
                        action = self.choose_action()

                # Perform the chosen action
                new_state, reward, done = self.env.step(
                    action, play_audio=self.play_audio, show_room=self.show_room
                )

                if self.plot_reward_vs_steps:
                    temp_rewards.append(reward)

                # Perform Update
                self.update()

                # store SARS in buffer
                if prev_state is not None and new_state is not None and not done:
                    self.dataset.write_buffer_data(
                        prev_state, action, reward, new_state, episode, step
                    )

                # Terminate the episode if done
                if done:
                    # terminal state is silence
                    silence_array = np.zeros_like(prev_state.audio_data)
                    terminal_silent_state = prev_state.make_copy_with_audio_data(audio_data=silence_array)
                    self.dataset.write_buffer_data(
                        prev_state, action, reward, terminal_silent_state, episode, step
                    )

                    end = time.time()
                    total_time = end - start

                    logging_str = (
                        f"\n\n"
                        f"Episode Summary \n"
                        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n"
                        f"- Episode: {episode+1}\n"
                        f"- Done a step: {step+1}\n"
                        f"- Time taken:   {total_time:04f} \n"
                        f"- Steps/Second: {float(step+1)/total_time:04f} \n"
                        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n"
                    )
                    logging.info(logging_str)
                    if self.plot_reward_vs_steps:
                        rewards_per_episode.append(temp_rewards)
                    break

                if step == (self.max_steps - 1) and self.plot_reward_vs_steps:
                    rewards_per_episode.append(temp_rewards)

                prev_state = new_state

            # Decay the epsilon
            self.epsilon = constants.MIN_EPSILON + (
                constants.MAX_EPSILON - constants.MIN_EPSILON
            ) * np.exp(-self.decay_rate * episode)

        if self.track_dist_vs_steps:
            utils.log_dist_and_num_steps(init_dist_to_target, steps_to_completion)
            # NOTE: plot will fail if agent doesn't find all sources within given number of steps
            utils.plot_dist_and_steps()
        
        if self.plot_reward_vs_steps:
            utils.log_reward_vs_steps(rewards_per_episode)
            utils.plot_reward_vs_steps()

    def choose_action(self):
        """
        This function must be implemented by whatever class inherits AgentBase.
        It will choose the action to take at any time step.

        Returns:
            action (int): the action to take
        """
        raise NotImplementedError()

    def update(self):
        """
        This function must be implemented by whatever class inherits AgentBase.
        It will perform an update (e.g. updating Q table or Q network)
        """
        raise NotImplementedError()


class RLAgent(AgentBase):
    def choose_action(self):
        """
        TODO
        """
        pass

    def update(self):
        """
        TODO
        """
        pass


class RandomAgent(AgentBase):
    def choose_action(self):
        """
        Since this is a random agent, we just randomly sample our action 
        every time.
        """
        return self.env.action_space.sample()

    def update(self):
        """
        No update for a random agent
        """
        pass

