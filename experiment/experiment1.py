import sys
sys.path.append("../src/")
from datetime import datetime
import os

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging 
from torch.utils.tensorboard import SummaryWriter

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

import warnings
warnings.filterwarnings("ignore")

"""
One of our main experiments for OtoWorld introductory paper 
"""

# Shoebox Room
nussl.utils.seed(0)
room = room_types.ShoeBox(x_length=8, y_length=8)

# Uncomment for Polygon Room
#room = room_types.Polygon(n=6, r=2, x_center=5, y_center=5)


source_folders_dict = {'../sounds/phone/': 1,
                        '../sounds/siren/': 1}

# Set up the gym environment
env = gym.make(
    "audio-room-v0",
    room_config=room.generate(),
    source_folders_dict=source_folders_dict,
    corners=room.corners,
    max_order=10,
    step_size=.5,
    acceptable_radius=1.0,
    absorption=1.0,
)
env.seed(0)

# create buffer data folders
utils.create_buffer_data_folders()

# fixing lengths
tfm = transforms.Compose([
    transforms.GetAudio(mix_key=['prev_state', 'new_state']),
    transforms.ToSeparationModel(),
    transforms.GetExcerpt(excerpt_length=32000,
                            tf_keys=['mix_audio_prev_state'], time_dim=1),
    transforms.GetExcerpt(excerpt_length=32000,
                            tf_keys=['mix_audio_new_state'], time_dim=1)
])

# create dataset object (subclass of nussl.datasets.BaseDataset)
dataset = BufferData(
    folder=constants.DIR_DATASET_ITEMS, 
    to_disk=True, 
    transform=tfm
)

# define tensorboard writer, name the experiment!
exp_name = 'new-exp-150eps'
exp_id = '{}_{}'.format(exp_name, datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
writer = SummaryWriter('runs/{}'.format(exp_id))


# Define the relevant dictionaries
env_config = {
    'env': env, 
    'dataset': dataset, 
    'episodes': 150, 
    'max_steps': 1000,
    'stable_update_freq': 150,
    'save_freq': 1, 
    'play_audio': False,
    'show_room': False,
    'writer': writer,
    'dense': True,
    'decay_rate': 0.0002,  # trial and error
    'decay_per_ep': True
}

save_path = os.path.join(constants.MODEL_SAVE_PATH, exp_name)
dataset_config = {
    'batch_size': 10, 
    'num_updates': 2, 
    'save_path': save_path
}

# clear save_path folder for each experiment
utils.clear_models_folder(save_path)

rnn_config = {
    'bidirectional': True,
    'dropout': 0.3,
    'filter_length': 256,
    'hidden_size': 50,
    'hop_length': 64,
    'mask_activation': ['softmax'],
    'mask_complex': False,
    'mix_key': 'mix_audio',
    'normalization_class': 'BatchNorm',
    'num_audio_channels': 1,
    'num_filters': 256,
    'num_layers': 1,
    'num_sources': 2,
    'rnn_type': 'lstm',
    'window_type': 'sqrt_hann',
}

stft_config = {
    'hop_length': 64,
    'num_filters': 256,
    'direction': 'transform',
    'window_type': 'sqrt_hann'
}

rnn_agent = RnnAgent(
    env_config=env_config, 
    dataset_config=dataset_config,
    rnn_config=rnn_config,
    stft_config=stft_config,
    learning_rate=.001,
    pretrained=True
)
torch.autograd.set_detect_anomaly(True)
rnn_agent.fit()

