import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import room_types
import agent
import audio_room
import utils
import constants
import nussl
from datasets import BufferData
import time
import audio_processing
from models import RnnQNet

"""
Experiment 3 details: 
Train the agent on the actual model which includes separation model + Q Network (The RNNQNet model) 
"""

def run():
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

    # create buffer data folders
    utils.create_buffer_data_folders()

    tfm = nussl.datasets.transforms.Compose([
        nussl.datasets.transforms.GetAudio(mix_key='new_state'),
        nussl.datasets.transforms.ToSeparationModel(),
        nussl.datasets.transforms.GetExcerpt(excerpt_length=32000, tf_keys=['mix_audio'], time_dim=1),
    ])

    # create dataset object (subclass of nussl.datasets.BaseDataset)
    dataset = BufferData(folder=constants.DIR_DATASET_ITEMS, to_disk=False, transform=tfm)

    # Define the relevant dictionaries
    env_config = {'env': env, 'dataset': dataset, 'episodes': 5, 'steps': 100, 'plot_reward_vs_steps': False}
    dataset_config = {'batch_size': 25, 'num_updates': 1}
    rnn_agent = RnnQNet(env_config=env_config, dataset_config=dataset_config)

    rnn_agent.fit()


if __name__ == '__main__':
    run()