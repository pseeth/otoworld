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

# setup logging (with different logger than the agent logger)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler = logging.FileHandler('agent.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info('\n')
logger.info('-'*50)
logger.info('\nStarting to Fit with Agent\n')
logger.info('-'*50)
logger.info('\n')


class AgentBase:
    def __init__(
        self,
        env,
        dataset,
        episodes=1,
        max_steps=100,
        gamma=0.98,
        alpha=0.001,
        decay_rate=0.0005,
        stable_update_freq=-1,
        save_freq=1,
        play_audio=False,
        show_room=False,
        writer=None,
        dense=True,
        decay_per_ep=False,
        decay_steps=150
    ):
        """
        This class is a base agent class which will be inherited when creating various agents.

        Args:
            self.env (gym object): The gym self.environment object which the agent is going to explore
            dataset (nussl dataset): Nussl dataset object for experience replay
            episodes (int): # of episodes to simulate
            max_steps (int): # of steps the agent can take before stopping an episode
            gamma (float): Discount factor
            alpha (float): Learning rate alpha
            decay_rate (float): decay rate for exploration rate (we want to decrease exploration as time proceeds)
            stable_update_freq (int): Update frequency value for stable networks (Target networks)
            play_audio (bool): choose to play audio at each iteration
            show_room (bool): choose to display the configurations and movements within a room
            writer (torch.utils.tensorboard.SummaryWriter): for logging to tensorboard
            dense (bool): makes the rewards more dense, less sparse
                gives reward for distance to closest source
        """
        self.env = env
        self.dataset = dataset
        self.episodes = episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = constants.MAX_EPSILON
        self.decay_rate = decay_rate
        self.stable_update_freq = stable_update_freq
        self.save_freq = save_freq
        self.play_audio = play_audio
        self.show_room = show_room
        self.writer = writer
        self.dense = dense
        self.losses = []
        self.cumulative_reward = 0
        self.total_experiment_steps = 0
        self.mean_episode_reward = []
        self.decay_per_ep = decay_per_ep
        self.decay_steps = decay_steps

    def fit(self):
        for episode in range(self.episodes):
            # Reset the self.environment and any other variables at beginning of each episode
            prev_state = None

            episode_rewards = []

            # Measure time to complete the episode
            start = time.time()
            for step in range(self.max_steps):
                self.total_experiment_steps += 1

                # log distance to sources to tensorboad
                # TODO: handle > 2 sources
                if len(self.env.source_locs) == 2:
                    self.writer.add_scalar('Distance/dist_to_source0', euclidean(self.env.source_locs[0], self.env.agent_loc), self.total_experiment_steps)
                    self.writer.add_scalar('Distance/dist_to_source1', euclidean(self.env.source_locs[1], self.env.agent_loc), self.total_experiment_steps)
                elif len(self.env.source_locs) == 1:
                    # TODO: refactor, super hacky right now
                    # currently don't know which source is remaining, 0 or 1 (there's just a source in the list without an identity)
                    remaining_source_path = self.env.direct_sources[0]
                    remaining_source = self.env.source_locs[0]
                    if constants.DIR_MALE in remaining_source_path:  # 1st source
                        self.writer.add_scalar('Distance/dist_to_source0', euclidean(remaining_source, self.env.agent_loc), self.total_experiment_steps)
                        self.writer.add_scalar('Distance/dist_to_source1', 0, self.total_experiment_steps)
                    else:
                        # 2nd source
                        self.writer.add_scalar('Distance/dist_to_source0', 0, self.total_experiment_steps)
                        self.writer.add_scalar('Distance/dist_to_source1', euclidean(remaining_source, self.env.agent_loc), self.total_experiment_steps)

                # Perform random actions with prob < epsilon
                #print('epsilon', self.epsilon)
                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    # For the first two steps (We don't have prev_state, new_state pair), then perform a random action
                    if step < 2:
                        action = self.env.action_space.sample()
                    else:
                        # This is where agent will actually do something
                        action = self.choose_action()

                # Perform the chosen action (NOTE: reward is a dictionary)
                new_state, reward, done = self.env.step(
                    action, play_audio=self.play_audio, show_room=self.show_room
                )

                # dense vs sparse 
                total_step_reward = 0
                if self.dense:
                    total_step_reward += sum(reward.values())
                else:
                    total_step_reward += (reward['step_penalty'] + reward['turn_off_reward'])
                        
                # record reward stats
                self.cumulative_reward += total_step_reward
                episode_rewards.append(total_step_reward)

                if reward['turn_off_reward'] == constants.TURN_OFF_REWARD:
                    print('In FIT. Received reward: {} at step {}\n'.format(total_step_reward, step))
                    logger.info(f"In FIT. Received reward {total_step_reward} at step: {step}\n")

                # Perform Update
                self.update()

                # store SARS in buffer
                if prev_state is not None and new_state is not None and not done:
                    self.dataset.write_buffer_data(
                        prev_state, action, total_step_reward, new_state, episode, step
                    )

                # Decay epsilon based on total steps (across all episodes, not within an episode)
                if not self.decay_per_ep:
                    self.epsilon = constants.MIN_EPSILON + (
                        constants.MAX_EPSILON - constants.MIN_EPSILON
                    ) * np.exp(-self.decay_rate * self.total_experiment_steps)
                    if self.total_experiment_steps % 200 == 0:
                        print("Epsilon decayed to {} at step {} ".format(self.epsilon, self.total_experiment_steps))

                # Update stable networks based on number of steps
                if step % self.stable_update_freq == 0:
                    self.update_stable_networks()

                # Terminate the episode if done
                if done:
                    # terminal state is silence
                    silence_array = np.zeros_like(prev_state.audio_data)
                    terminal_silent_state = prev_state.make_copy_with_audio_data(audio_data=silence_array)
                    self.dataset.write_buffer_data(
                        prev_state, action, total_step_reward, terminal_silent_state, episode, step
                    )

                    # record mean reward for this episode
                    self.mean_episode_reward = np.mean(episode_rewards)
                    self.writer.add_scalar('Reward/mean_per_episode', self.mean_episode_reward, episode)
                    self.writer.add_scalar('Reward/cumulative', self.cumulative_reward, self.total_experiment_steps)

                    end = time.time()
                    total_time = end - start

                    # log episode summary
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
                    print(logging_str)
                    logger.info(logging_str)

                    # break and go to new episode if done
                    break

                prev_state = new_state

            # if episode % self.stable_update_freq == 0:
            #     self.update_stable_networks()

            if episode % self.save_freq == 0:
                name = 'ep{}.pt'.format(episode)
                self.save_model(name)

            # Decay the epsilon
            if self.decay_per_ep:
                self.epsilon = constants.MIN_EPSILON + (
                    constants.MAX_EPSILON - constants.MIN_EPSILON
                ) * np.exp(-self.decay_rate * (episode + 1))

            # Reset the environment
            self.env.reset()

    def choose_action(self):
        """
        This function must be implemented by subclass.
        It will choose the action to take at any time step.

        Returns:
            action (int): the action to take
        """
        raise NotImplementedError()

    def update(self):
        """
        This function must be implemented by subclass.
        It will perform an update (e.g. updating Q table or Q network)
        """
        raise NotImplementedError()

    def update_stable_networks(self):
        """
            This function must be implemented by subclass.
            It will perform an update to the stable networks. I.E Copy values from current network to target network
            """
        raise NotImplementedError()

    def save_model(self, name):
        raise NotImplementedError()


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

    def update_stable_networks(self):
        pass

    def save_model(self, name):
        pass

