import gym
import numpy as np
import matplotlib.pyplot as plt

# get modules from diff folder (src folder)
import sys

sys.path.append("../src/")

import room_types
import agent
import audio_room
import utils


def run_rl_agent():
    # paths of audio files
    paths = utils.choose_random_files()

    # Shoebox Room
    room = room_types.ShoeBox(x_length=10, y_length=10)

    # Uncomment for Polygon Room
    # room = room_types.Polygon(n=6, r=2, x_center=5, y_center=5)

    agent_loc = np.array([3, 8])

    # Set up the gym environment
    env = gym.make(
        "audio-room-v0",
        room_config=room.generate(),
        agent_loc=agent_loc,
        corners=room.corners,
        max_order=10,
        step_size=1.0,
        direct_sources=paths,
        acceptable_radius=0.8,
    )
    env.add_sources()

    # Load the agent class
    a = agent.RandomAgent(episodes=10, steps=1000)
    a.fit(env)


if __name__ == "__main__":
    run_rl_agent()
