import sys
sys.path.append("../src/")  # get modules from src folder

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
    dataset = BufferData(folder=constants.DIR_DATASET_ITEMS, to_disk=True, transform=tfm)

    # Load the agent class
    a = agent.RandomAgent(env=env, dataset=dataset, episodes=5, steps=100, plot_reward_vs_steps=False)
    a.fit()

    # print(dataset[0])
    # print((dataset.items))
    print("Buffer filled: ", len(dataset.items))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=False)

    # for index, item in enumerate(dataloader):
    #     for key in item:
    #         print(key, item[key].shape)
    #     break

    """
    # Parameters for build_recurrent_end_to_end:
    # ==============================================================================
    build_recurrent_end_to_end.bidirectional = True
    build_recurrent_end_to_end.dropout = 0.3
    build_recurrent_end_to_end.filter_length = 256
    build_recurrent_end_to_end.hidden_size = 600
    build_recurrent_end_to_end.hop_length = 64
    build_recurrent_end_to_end.mask_activation = ['sigmoid']
    build_recurrent_end_to_end.mask_complex = False
    build_recurrent_end_to_end.mix_key = 'mix_audio'
    build_recurrent_end_to_end.normalization_class = 'BatchNorm'
    build_recurrent_end_to_end.num_audio_channels = 1 # or 2?
    build_recurrent_end_to_end.num_filters = 256
    build_recurrent_end_to_end.num_layers = 4
    build_recurrent_end_to_end.num_sources = 2
    build_recurrent_end_to_end.rnn_type = 'lstm'
    build_recurrent_end_to_end.trainable = False
    build_recurrent_end_to_end.window_type = 'sqrt_hann'
    """
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



"""
Timing experiment results: 

Setting: 
Episodes = 10 
Steps per episode: 1000
Buffer size: 5000 
-------------------------------------------------
With to_disk = true 
Buffer Filled: 2499
Total time taken: 45.95445275306702 seconds 

Buffer filled:  1995
Total time taken: 31.331244230270386 seconds

Buffer filled:  3973
Total time taken: 96.270427942276 seconds

Buffer filled:  3054
Total time taken: 36.36921525001526 seconds

Buffer filled:  2121
Total time taken: 32.37190365791321 seconds
-------------------------------------------------
With to_disk = False 
Buffer filled:  5000
Total time taken: 52.56600522994995 seconds

Buffer filled:  5000
Total time taken: 47.749303340911865 seconds

Buffer filled:  3444
Total time taken: 33.542359828948975 seconds

Buffer filled:  3267
Total time taken: 28.833098888397217 seconds

---

Setting: 
Episodes = 10 
Steps per episode: 1000
Buffer size: 500  (smaller buffer so should be full more often)
-------------------------------------------------
With to_disk = True 

Buffer filled:  500
Total time taken: 54.17424130439758 seconds

Buffer filled:  500
Total time taken: 56.30148649215698 seconds

Buffer filled:  500
Total time taken: 39.96235132217407 seconds
-------------------------------------------------
With to_disk = False 

Buffer filled:  500
Total time taken: 37.481664419174194 seconds

Buffer filled:  500
Total time taken: 44.5657320022583 seconds

Buffer filled:  500
Total time taken: 39.662904500961304 seconds
"""