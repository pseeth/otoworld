import sys
sys.path.append("../src")

from pyroomacoustics import ShoeBox, Room


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
from transforms import GetExcerpt

"""
Takes too long to run and isn't a great test. Assertion will be added in GetExcerpt to ensure
the mixes for the prev and new state are the same length
"""

def test_mix_lengths():
    pass
#     # Shoebox Room
#     room = room_types.ShoeBox(x_length=10, y_length=10)

#     # Uncomment for Polygon Room
#     # room = room_types.Polygon(n=6, r=2, x_center=5, y_center=5)

#     agent_loc = np.array([3, 8])

#     # Set up the gym environment
#     env = gym.make(
#         "audio-room-v0",
#         room_config=room.generate(),
#         agent_loc=agent_loc,
#         corners=room.corners,
#         max_order=10,
#         step_size=1.0,
#         acceptable_radius=0.8,
#     )

#     # create buffer data folders
#     utils.create_buffer_data_folders()

#     tfm = transforms.Compose([
#         transforms.GetAudio(mix_key=['prev_state', 'new_state']),
#         transforms.ToSeparationModel(),
#         transforms.GetExcerpt(excerpt_length=32000,
#                               tf_keys=['mix_audio_prev_state', 'mix_audio_new_state'], time_dim=1),
#     ])

#     dataset = BufferData(folder=constants.DIR_DATASET_ITEMS, to_disk=False, transform=tfm)

#     # run really short experiment to generate data
#     # Define the relevant dictionaries
#     env_config = {'env': env, 'dataset': dataset, 'episodes': 3, 'max_steps': 3, 'plot_reward_vs_steps': False,
#                   'stable_update_freq': 3, 'epsilon': 0.7, 'save_freq': 1}
#     dataset_config = {'batch_size': 3, 'num_updates': 2, 'save_path': '../models/'}
#     rnn_agent = RnnAgent(env_config=env_config, dataset_config=dataset_config)

#     rnn_agent.fit()

#     # test mix lengths, want to be equal (THIS MAY TAKE A WHILE)
#     # this test is not perfect, may get lucky and have all dataset items be same length, but it's already expensive to compute 
#     for t in tfm.transforms:
#         lengths = []
#         print(len(dataset))
#         for i in range(len(dataset)):
#             if isinstance(t, GetExcerpt):
#                 data = t(dataset[i])
#                 for k, v in data.items():
#                     if k in t.time_frequency_keys:
#                         lengths.append(v.size())
#                 print(lengths)
#                 assert(lengths[0] == lengths[1])
            
# test_mix_lengths()