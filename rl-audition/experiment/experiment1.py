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


def run_random_agent():
    # paths of audio files
    paths = utils.choose_random_files()

    # Shoebox Room
    room = room_types.ShoeBox(x_length=5, y_length=5)

    # Uncomment for Polygon Room
    # room = room_types.Polygon(n=6, r=2, x_center=5, y_center=5)

    agent_loc = np.array([3, 3])

    # Set up the gym environment
    env = gym.make(
        "audio-room-v0",
        room_config=room.generate(),
        agent_loc=agent_loc,
        corners=room.corners,
        max_order=10,
        step_size=1.0,
        direct_sources=paths,
        acceptable_radius=1.0
    )

    # create buffer data folders
    utils.create_buffer_data_folders()

    tfm = nussl.datasets.transforms.Compose([
        nussl.datasets.transforms.GetAudio(mix_key='new_state'),
        nussl.datasets.transforms.ToSeparationModel(),
        nussl.datasets.transforms.GetExcerpt(excerpt_length=32000, tf_keys=['mix_audio'], time_dim=1),
    ])

    # create dataset object (subclass of nussl.datasets.BaseDataset)
    dataset = BufferData(folder=constants.DIR_DATASET_ITEMS, to_disk=True, transform=tfm)

    # Load the agent class
    a = agent.RandomAgent(env=env, dataset=dataset, episodes=3, steps=200, plot_reward_vs_steps=False)
    a.fit()

    # print(dataset[0])
    print("Buffer filled: ", len(dataset.items))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=False)

    # Parameters for build_recurrent_end_to_end:
    config = nussl.ml.networks.builders.build_recurrent_end_to_end(
        bidirectional=True, dropout=0.3, filter_length=256, hidden_size=300, hop_length=64, mask_activation=['sigmoid'],
        mask_complex=False, mix_key='mix_audio', normalization_class='BatchNorm', num_audio_channels=1, num_filters=256,
        num_layers=2, num_sources=2, rnn_type='lstm', trainable=False, window_type='sqrt_hann'
    )
    
    model = nussl.ml.SeparationModel(config)
    print(model)


if __name__ == "__main__":
    # runs = 3
    # for i in range(runs):
    #     print("Run: {}".format(i+1))
    #     start = time.time()
    #     run_random_agent()
    #     end = time.time()
    #     print("Total time taken: {} seconds".format(end-start))
    run_random_agent()
