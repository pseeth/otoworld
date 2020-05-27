# NOTE: to run, need to cd into tests/, then run pytest
import sys
sys.path.append("../src")

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
from agent import RandomAgent


def test_experiment_shoebox():
    """
    Testing a run with ShoeBox room

    TODO
    """
    # paths of audio files
    paths = utils.choose_random_files()

    # Shoebox Room
    room = room_types.ShoeBox(x_length=10, y_length=10)

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
    a = agent.RandomAgent(env=env, dataset=dataset, episodes=2, max_steps=10, plot_reward_vs_steps=False)
    a.fit()

    # what should we assert? 
    #assert()


def test_experiment_polygon():
    """
    Testing a run with Polygon room

    TODO
    """
    # Uncomment for Polygon Room
    room = room_types.Polygon(n=6, r=2, x_center=5, y_center=5)
