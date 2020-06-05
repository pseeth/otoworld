import gym
import numpy as np
import torch
import logging
import agent
import utils
import constants
import nussl
import audio_processing
import agent
from datasets import BufferData, RLDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# setup logging (with different logger than the agent logger)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler = logging.FileHandler('model.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class RnnAgent(agent.AgentBase):
    def __init__(self, env_config, dataset_config, rnn_config=None, stft_config=None, verbose=False):
        """
        Args:
            env_config (dict): Dictionary containing the audio environment config
            dataset_config (dict): Dictionary consisting of dataset related parameters. List of parameters
            'batch_size' : Denotes the batch size of the samples
            'num_updates': Amount of iterations we run the training for in each pass.
             Ex - If num_updates = 5 and batch_size = 25, then we run the update process 5 times where in each run we
             sample 25 data points.
             'sampler': The sampler to use. Ex - Weighted Sampler, Batch sampler etc

            rnn_config (dict):  Dictionary containing the parameters for the model
            stft_config (dict):  Dictionary containing the parameters for STFT
        """

        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('DEVICE:', self.device)
        # Use default config if configs are not provided by user
        if rnn_config is None:
            self.rnn_config = nussl.ml.networks.builders.build_recurrent_end_to_end(
        bidirectional=True, dropout=0.3, filter_length=256, hidden_size=300, hop_length=64, mask_activation=['sigmoid'],
        mask_complex=False, mix_key='mix_audio', normalization_class='BatchNorm', num_audio_channels=1, num_filters=256,
        num_layers=2, num_sources=2, rnn_type='lstm', trainable=False, window_type='sqrt_hann')
        else:
            self.rnn_config = nussl.ml.networks.builders.build_recurrent_end_to_end(**rnn_config)

        if stft_config is None:
            self.stft_diff = nussl.ml.networks.modules.STFT(hop_length=128, filter_length=512, direction='transform',
                                           num_filters=512)
        else:
            self.stft_diff = nussl.ml.networks.modules.STFT(**stft_config)

        # Initialize the Agent Base class
        super().__init__(**env_config)

        # Uncomment this to find backprop errors
        # torch.autograd.set_detect_anomaly(True)

        # Initialize the rnn model
        # self.rnn_model = nussl.ml.SeparationModel(self.rnn_config, verbose=verbose)
        self.rnn_model = RnnSeparator(self.rnn_config).to(self.device)
        self.rnn_model_stable = RnnSeparator(self.rnn_config).to(self.device)  # Fixed Q network

        # Initialize dataset related parameters
        self.bs = dataset_config['batch_size']
        self.num_updates = dataset_config['num_updates']
        self.dynamic_dataset = RLDataset(buffer=self.dataset, sample_size=self.bs)
        self.dataloader = torch.utils.data.DataLoader(self.dynamic_dataset, batch_size=self.bs)

        # Initialize network layers for DQN network
        filter_length = stft_config['filter_length']+2 if stft_config is not None else 514
        total_actions = self.env.action_space.n
        network_params = {'filter_length': filter_length, 'total_actions': total_actions, 'stft_diff': self.stft_diff}
        self.q_net = DQN(network_params).to(self.device)
        self.q_net_stable = DQN(network_params).to(self.device)  # Fixed Q net

        params = list(self.rnn_model.parameters()) + list(self.q_net.parameters())
        self.optimizer = optim.Adam(params, lr=0.001)

        # Folder path where the model will be saved
        self.SAVE_PATH = dataset_config['save_path']

    def update(self):
        """
        Args:
            episode (int): Current episode number to keep update the stable Q networks every k episodes

        Returns:
        """
        # print("Size of dataset {}".format(len(self.dataset.items)) )#len(self.dynamic_dataset.buffer)))
        # Run the update only if samples >= batch_size
        if len(self.dataset.items) < self.bs:
            return
        #
        # # Take the buffer data and make it into torch data loader object
        # dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.bs, shuffle=True)

        for index, data in enumerate(self.dataloader):
            if index > self.num_updates:
                break
            # print("Action ", data['action'].shape)
            # Get the total number of time steps
            total_time_steps = data['mix_audio_prev_state'].shape[-1]
            # print("Data shape: ", data['mix_audio_prev_state'].shape)
            # Reshape the mixture to pass through the separation model (Convert dual channels into one)
            # Also, rename the state to work on to mix_audio so that it can pass through remaining nussl architecture
            # Move, everything to GPU
            data['mix_audio'] = data['mix_audio_prev_state'].float().view(-1, 1, total_time_steps).to(self.device)
            data['action'] = data['action'].to(self.device)
            data['reward'] = data['reward'].to(self.device)
            # Get the separated sources by running through RNN separation model

            output = self.rnn_model(data)
            # Pass then through the DQN model to get q values
            q_values = self.q_net(output, total_time_steps).gather(1, data['action'])
            # print("Q values", q_values.shape)

            with torch.no_grad():
                # Now, get Q-values for the next-state
                # Get the total number of time steps
                total_time_steps = data['mix_audio_new_state'].shape[-1]
                # print("Data shape: ", data['mix_audio_new_state'].shape)
                # Reshape the mixture to pass through the separation model (Convert dual channels into one)
                data['mix_audio'] = data['mix_audio_new_state'].float().view(-1, 1, total_time_steps).to(self.device)
                output = self.rnn_model_stable(data)
                q_values_next = self.q_net_stable(output, total_time_steps).max(1)[0].unsqueeze(-1)
                # print("Next state", q_values_next.shape)

            expected_q_values = data['reward'] + self.gamma*q_values_next
            # Calculate loss
            loss = F.mse_loss(q_values, expected_q_values)
            self.losses.append(loss)
            #print("Loss:", loss)
            logger.info(f"Loss: {loss}")
            self.writer.add_scalar('Loss/train', loss, len(self.losses))
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def choose_action(self):
        with torch.no_grad():
            # Get the latest state from the buffer
            # print("Last ptr: ", self.dataset.last_ptr, "len: ", len(self.dataset.items))
            # print("Value at that ptr: ", self.dataset[self.dataset.last_ptr])
            data = self.dataset[self.dataset.last_ptr]
            # Perform the forward pass
            total_time_steps = data['mix_audio_new_state'].shape[-1]
            data['mix_audio'] = data['mix_audio_new_state'].float().view(-1, 1, total_time_steps).to(self.device)
            output = self.rnn_model(data)
            q_value = self.q_net(output, total_time_steps).max(1)[0].unsqueeze(-1)

            q_value = q_value[0].item()

            if math.isnan(q_value):
                q_value = self.env.action_space.sample()
                logging_str = (
                        f"\n"
                        f"Received NaN Q Value \n"
                        f"- Data: {data}\n"
                        f"- Output RNN: {output}\n"
                        f"- Last ptr:   {self.dataset.last_ptr} \n"
                        f"- Q Value: {q_value} \n\n"
                    )
                logger.info(logging_str)
            else:
                q_value = int(q_value)

            return q_value

    def update_stable_networks(self):
        print("Target network updated!")
        self.rnn_model_stable.load_state_dict(self.rnn_model.state_dict())
        self.q_net_stable.load_state_dict(self.q_net.state_dict())

    def save_model(self, name):
        """
        Args:
            name (str): Name contains the episode information (To give saved models unique names)
        """
        # Save the parameters for rnn model and q net separately
        torch.save(self.rnn_model.state_dict(), self.SAVE_PATH + 'rnn_' + name)
        torch.save(self.q_net.state_dict(), self.SAVE_PATH + 'qnet_' + name)


class RnnSeparator(nn.Module):
    def __init__(self, rnn_config, verbose=False):
        super(RnnSeparator, self).__init__()
        self.rnn_model = nussl.ml.SeparationModel(rnn_config, verbose=verbose)

    def forward(self, x):
        return self.rnn_model(x)


class DQN(nn.Module):
    def __init__(self, network_params):
        """

        Args:
            network_params (dict): Dict of network parameters
        """
        super(DQN, self).__init__()

        self.stft_diff = network_params['stft_diff']
        self.fc1 = nn.Linear(network_params['filter_length'] * 2, 64)
        self.fc2 = nn.Linear(64, network_params['total_actions'])

    def forward(self, output, total_time_steps):
        # Reshape the output again to get dual channels
        output['audio'] = output['audio'].view(-1, 2, total_time_steps, 2)

        # Perform short time fourier transform of this output
        stft_data = self.stft_diff(output['audio'], direction='transform')

        # Get the IPD and ILD features from the stft data
        ipd, ild = audio_processing.ipd_ild_features(stft_data)

        # print("IPD: {}, ILD {}".format(ipd.shape, ild.shape))

        # Concatenate IPD And ILD features
        X = torch.cat((ipd, ild), dim=2)  # Concatenate the features dimension

        # Sum over the time frame axis
        X = torch.sum(X, dim=1)

        # Flatten the features
        num_features = self.flatten_features(X)
        X = X.view(-1, num_features)  # Shape = [Batch_size, filter_length*2]

        # Run through simple feedforward network
        X = F.relu(self.fc1(X))
        q_values = F.softmax(self.fc2(X), dim=1)

        return q_values

    def flatten_features(self, x):
        size = x.size()[1:]  # Flatten all dimensions except the batch
        num_features = 1
        for dimension in size:
            num_features *= dimension

        return num_features








