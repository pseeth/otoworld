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

"""
Experiment 2 details: 
Shoddy code to see if separation model works  

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

    tfm = nussl.datasets.transforms.Compose([
        nussl.datasets.transforms.GetAudio(mix_key='new_state'),
        nussl.datasets.transforms.ToSeparationModel(),
        nussl.datasets.transforms.GetExcerpt(excerpt_length=32000, tf_keys=['mix_audio'], time_dim=1),
    ])

    # create dataset object (subclass of nussl.datasets.BaseDataset)
    dataset = BufferData(folder=constants.DIR_DATASET_ITEMS, to_disk=False, transform=tfm)

    # Load the agent class
    a = agent.RandomAgent(env=env, dataset=dataset, episodes=5, max_steps=100, plot_reward_vs_steps=False)
    a.fit()

    print("Buffer filled: ", len(dataset.items))

    dataloader = dataset
    #
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=False)
    #
    # Parameters for build_recurrent_end_to_end:
    config = nussl.ml.networks.builders.build_recurrent_end_to_end(
        bidirectional=True, dropout=0.3, filter_length=256, hidden_size=300, hop_length=64, mask_activation=['sigmoid'],
        mask_complex=False, mix_key='mix_audio', normalization_class='BatchNorm', num_audio_channels=1, num_filters=256,
        num_layers=2, num_sources=2, rnn_type='lstm', trainable=False, window_type='sqrt_hann'
    )

    model = nussl.ml.SeparationModel(config, verbose=True)
    print(config)
    print(model)

    stft_diff = nussl.ml.networks.modules.STFT(hop_length=128, filter_length=512, direction='transform',
                                               num_filters=512)
    #
    for i in range(2):
        for index, data in enumerate(dataloader):
            # print(data['mix_audio'].shape)
            print(data.keys())
            print("Original shape: ", data['mix_audio'].shape)
            data['mix_audio'] = data['mix_audio'].float().view(-1, 1, 32000)
            print(data['mix_audio'].shape)
            # print((data['mix_audio']).float())
            # output = model({'mix_audio': data['mix_audio']})
            output = model(data)
            print(output.keys())
            print(output['audio'].shape, output['mask'].shape)
            output['audio'] = output['audio'].view(-1, 2, 32000, 2)
            # Shape of output['audio'] = [batch_size, channels, total_time_steps, sources]

            print("OG Audio", output['audio'].shape)

            new_output = stft_diff(output['audio'], direction='transform')
            # Shape of new_output = [batch_size, time_frames, mag+phase, channels, sources]
            # Here, time_frames = total_time_steps/hop_length = 32000/128 = 250 + 1 = 251
            # Mag+phase dimensionality comes from the filter_length size. Filter_length + 2 = 512 + 2
            ipd, ild = audio_processing.ipd_ild_features(new_output)
            print("IPD, ILD", ipd.shape, ild.shape)

            if index > 1:
                break

        # print(output.shape)

    # print(dataset[5].keys())
    # for batch in range(100):
    #     # print(data['mix_audio'].shape)
    #     data = dataset[:25]
    #     print(data.keys())
    #     print("Original shape: ", data['mix_audio'].shape)
    #     data['mix_audio'] = data['mix_audio'].float().view(-1, 1, 32000)
    #     print(data['mix_audio'].shape)
    #     # print((data['mix_audio']).float())
    #     # output = model({'mix_audio': data['mix_audio']})
    #     output = model(data)
    #     print(output.keys())
    #     print(output['audio'].shape, output['mask'].shape)
    #     output['audio'] = output['audio'].view(-1, 2, 32000, 2)
    #     # Shape of output['audio'] = [batch_size, channels, total_time_steps, sources]
    #
    #     print("OG Audio", output['audio'].shape)
    #
    #     new_output = stft_diff(output['audio'], direction='transform')
    #     # Shape of new_output = [batch_size, time_frames, mag+phase, channels, sources]
    #     # Here, time_frames = total_time_steps/hop_length = 32000/128 = 250 + 1 = 251
    #     # Mag+phase dimensionality comes from the filter_length size. Filter_length + 2 = 512 + 2
    #     ipd, ild = audio_processing.ipd_ild_features(new_output)
    #     print("IPD, ILD", ipd.shape, ild.shape)
    #
    #     if batch > 1:
    #         break
    #
    #     # print(output.shape)


if __name__ == '__main__':
    run()
