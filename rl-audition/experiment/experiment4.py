import sys
sys.path.append("../src/")

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
from models import RnnAgent
import transforms

"""
Experiment 4 details: 
Make sure that the buffered data no longer contains time sequences of unequal lengths  
"""


def run():
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
        acceptable_radius=0.8,
    )

    # create buffer data folders
    utils.create_buffer_data_folders()

    # tfm = nussl.datasets.transforms.Compose([
    #     nussl.datasets.transforms.GetAudio(mix_key='new_state'),
    #     nussl.datasets.transforms.ToSeparationModel(),
    #     nussl.datasets.transforms.GetExcerpt(excerpt_length=32000, tf_keys=['mix_audio'], time_dim=1),
    # ])

    tfm = transforms.Compose([
        transforms.GetAudio(mix_key=['prev_state', 'new_state']),
        transforms.ToSeparationModel(),
        transforms.GetExcerpt(excerpt_length=32000,
                              tf_keys=['mix_audio_prev_state', 'mix_audio_new_state'], time_dim=1),
    ])

    # tfm = transforms.Compose([
    #     transforms.GetAudio(mix_key=['prev_state', 'new_state']),
    #     transforms.ToSeparationModel(),
    #     transforms.GetExcerpt(excerpt_length=32000,
    #                           tf_keys=['mix_audio_prev_state'], time_dim=1),
    #     transforms.GetExcerpt(excerpt_length=32000,
    #                           tf_keys=['mix_audio_new_state'], time_dim=1)
    # ])

    # create dataset object (subclass of nussl.datasets.BaseDataset)
    dataset = BufferData(folder=constants.DIR_DATASET_ITEMS, to_disk=False, transform=tfm)

    # # Define the relevant dictionaries
    # env_config = {'env': env, 'dataset': dataset, 'episodes': 10, 'max_steps': 500, 'plot_reward_vs_steps': False,
    #               'stable_update_freq': 3, 'epsilon': 0.7, 'save_freq': 1}
    # dataset_config = {'batch_size': 25, 'num_updates': 2, 'save_path': '../models/'}
    # rnn_agent = RnnAgent(env_config=env_config, dataset_config=dataset_config)
    #
    # rnn_agent.fit()

    a = agent.RandomAgent(env=env, dataset=dataset, episodes=10, max_steps=500, plot_reward_vs_steps=False)
    a.fit()
    # print("Buffer size: ", len(dataset))
    for index, data in enumerate(dataset):
        if data['mix_audio_prev_state'].shape[-1] != 32000 or data['mix_audio_new_state'].shape[-1] != 32000:
            print(index, data['mix_audio_prev_state'].shape, data['mix_audio_new_state'].shape)
            print(data)
            print(data['mix_audio_prev_state'])
            print("-----")
            print(data['mix_audio_new_state'])
            print("----")

        # print(data['mix_audio_prev_state'].shape)


if __name__ == '__main__':
    run()
