import numpy as np
import gym
import warnings
import time
from scipy.spatial.distance import euclidean
import utils
import constants
from collections import deque
import nussl

from dataset import BufferData

class AgentBase:
    def __init__(
        self,
        dataset,
        episodes=1,
        steps=10,
        blen=1000,
        gamma=0.9,
        alpha=0.001,
        epsilon=1.0,
        decay_rate=0.005,
        play_audio=False,
        show_room=False,
    ):
        """
        This class is a base agent class which will be inherited when creating various agents.

        Args:
            episodes (int): # of episodes to simulate
            steps (int): # of steps the agent can take before stopping an episode
            blen (int): # of entries which the replay buffer can store
            gamma (float): Discount factor
            alpha (float): Learning rate alpha
            epsilon (float): Exploration rate, P(taking random action)
            decay_rate (float): decay rate for exploration rate (we want to decrease exploration as time proceeds)
            play_audio (bool): choose to play audio at each iteration
            show_room (bool): choose to display the configurations and movements within a room
        """
        self.dataset = dataset
        self.episodes = episodes
        self.max_steps = steps
        self.blen = blen
        self.buffer = deque(maxlen=blen)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.play_audio = play_audio
        self.show_room = show_room
    
    def fit(self, env):
        # keep track of stats
        init_dist_to_target = []
        steps_to_completion = []

        for episode in range(self.episodes):
            # Reset the environment and any other variables at beginning of each episode
            env.reset()
            prev_state = None

            # Measure distance with the all sources
            init_dist_to_target.append(
                sum([euclidean(env.agent_loc, source) for source in env.source_locs]))
            steps_to_completion.append(0)
            
            # Measure time to complete the episode
            start = time.time()
            for step in range(self.max_steps):
                steps_to_completion[-1] += 1

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
                        action = self.choose_action(env)

                # Perform the chosen action
                new_state, reward, done = env.step(
                    action, play_audio=self.play_audio, show_room=self.show_room
                )

                # Perform the q-update or whatever we are using over here
                """
                Update q network 
                """

                # store SARS in buffer
                if prev_state is not None and new_state is not None and not done:
                    self.buffer.append((prev_state, action, reward, new_state))
                    utils.write_buffer_data(
                        prev_state, action, reward, new_state, episode, step, self.dataset
                    )

                # Terminate the episode if done
                if done:
                    # terminal state is silence
                    silence_array = np.zeros_like(prev_state.audio_data)
                    terminal_silent_state = prev_state.make_copy_with_audio_data(
                        audio_data=silence_array)
                    #self.buffer.append((prev_state, action, reward, terminal_silent_state))
                    utils.write_buffer_data(
                        prev_state, action, reward, terminal_silent_state, episode, step, self.dataset
                    )

                    end = time.time()
                    print("Done! at step ", step + 1)
                    print("Time: ", end - start, "seconds")
                    print("Steps/second: ", float(step + 1) / (end - start))
                    break

                prev_state = new_state

            # Decay the epsilon
            self.epsilon = constants.MIN_EPSILON + (
                constants.MAX_EPSILON - constants.MIN_EPSILON
            ) * np.exp(-self.decay_rate * episode)
        
        """
        Quick note about the plot: 
        It will fail to plot if the agent fails to find all sources within the given time 
        steps Deal with this later 
        """
        utils.log_dist_and_num_steps(init_dist_to_target, steps_to_completion)
        utils.plot_dist_and_steps()

    def choose_action(self, env):
        '''
        This function needs to be overwritten when implementing an agent. It 
        will choose the action to take at any given point.

        Args:
            env (gym): The gym environment you are training from
        '''
        pass


class RandomAgent(AgentBase):
    def choose_action(self, env):
        '''
        Since this is a random agent, we just randomly sample our action 
        every time.
        '''
        return env.action_space.sample()


class PerfectAgentORoom2(AgentBase):
    def choose_action(self, env):
        '''
        The action selection is simple, just find the closest point and move towards it.
        '''
        closest, distance = env.source_locs[0], euclidean(
            env.agent_loc, env.source_locs[0])
        for point in env.source_locs:
            temp = euclidean(env.agent_loc, point)
            if temp < distance:
                distance = temp
                closest = point

        prob = np.random.randn(1)
        if prob > 0.7:
            if env.agent_loc[0] < closest[0]:
                if env.room.is_inside([env.agent_loc[0] + 1, env.agent_loc[1]]):
                    action = 1
                    env.agent_loc[0] += 1
                elif (
                     env.room.is_inside(
                        [env.agent_loc[0], env.agent_loc[1] - 1])
                ):
                    action = 3
                    env.agent_loc[1] -= 1
                elif  env.room.is_inside([env.agent_loc[0], env.agent_loc[1] + 1]):
                    action = 2
                    env.agent_loc[1] += 1
            elif env.agent_loc[0] > closest[0]:
                if  env.room.is_inside([env.agent_loc[0] - 1, env.agent_loc[1]]):
                    action = 0
                    env.agent_loc[0] -= 1
                elif (
                     env.room.is_inside(
                        [env.agent_loc[0], env.agent_loc[1] - 1])
                ):
                    action = 3
                    env.agent_loc[1] -= 1
                elif  env.room.is_inside([env.agent_loc[0], env.agent_loc[1] + 1]):
                    action = 2
                    env.agent_loc[1] += 1

            elif env.agent_loc[0] == closest[0]:
                if env.agent_loc[1] < closest[1]:
                    action = 2
                    env.agent_loc[1] += 1
                else:
                    action = 3
                    env.agent_loc[1] -= 1
        else:
            action = np.random.randint(4, 6)
        return action


class HumanAgent:
    def __init__(
        self,
        target_loc,
        agent_loc,
        episodes=1,
        max_steps=50,
        converge_steps=10,
        step_size=None,
        acceptable_radius=1,
        play_audio=True,
        show_room=True,
    ):
        """
        This class is a human agent. The moves will be played by a human player. Easy way of navigating the environment
        ourselves for testing and debugging purposes.
        Args:
            target_loc (List[int] or np.array): the location of the target in the room
            agent_loc (List[int] or np.array): the initial location of the agent in the room
            episodes (int): # of episodes to simulate
            max_steps (int): # of steps the agent can take before stopping an episode
            converge_steps (int): # of steps the perfect agent should make before rewards
            acceptable_radius (float): radius of acceptable range the agent can be in to be considered done
            step_size (float): specified step size else we programmatically assign it
        """
        self.episodes = episodes
        closest = target_loc
        env.agent_loc = agent_loc
        self.play_audio = play_audio
        self.show_room = show_room

        self.max_steps = max_steps
        self.converge_steps = converge_steps
        self.acceptable_radius = acceptable_radius

        # Dictionary to convert 'wasd' to numbers
        self.key_to_action = {"w": 2, "a": 0, "s": 3, "d": 1}
        self.valid_actions = ["w", "a", "s", "d"]
        self.valid_angles = ["0", "1", "2"]

        # Finding the total distance to determine step size (total_dis / number of steps to converge)
        x_dis = abs(env.agent_loc[0] - closest[0])
        y_dis = abs(env.agent_loc[1] - closest[1])
        total_dis = x_dis + y_dis
        if step_size:
            self.step_size = step_size
        else:
            self.step_size = (total_dis) / self.converge_steps

    def fit(self, env):
        # ("Enter action (wasd) followed by orientation: (012)")
        """
        0 = Don't orient
        1 = Orient left 15 degrees
        2 = Orient right 15 degrees
        """
        done = False
        while not done:
            action, angle = map(str, input().split())

            if action in self.valid_actions and angle in self.valid_angles:
                new_state, closest, reward, done = env.step(
                    (self.key_to_action[action], int(angle)), self.play_audio, self.show_room,
                )
            else:
                # Pass some dummy action
                warnings.warn("Invalid action!")
                new_state, closest, reward, done = env.step(
                    (0, 0), self.play_audio, self.show_room
                )
