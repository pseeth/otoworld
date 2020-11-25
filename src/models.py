import os
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


class RnnAgent(agent.AgentBase):
    def __init__(self, env_config, dataset_config, rnn_config=None, stft_config=None, 
                 verbose=False, autoclip_percentile=10, learning_rate=.001, pretrained=False):
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
                bidirectional=True, dropout=0.3, filter_length=256, hidden_size=300, 
                hop_length=64, mask_activation=['sigmoid'], mask_complex=False, mix_key='mix_audio', 
                normalization_class='BatchNorm', num_audio_channels=1, num_filters=256,
                num_layers=2, num_sources=2, rnn_type='lstm', trainable=False, window_type='sqrt_hann'
            )
        else:
            self.rnn_config = nussl.ml.networks.builders.build_recurrent_end_to_end(**rnn_config)

        if stft_config is None:
            self.stft_diff = nussl.ml.networks.modules.STFT(
                hop_length=128, filter_length=512, 
                direction='transform', num_filters=512
            )
        else:
            self.stft_diff = nussl.ml.networks.modules.STFT(**stft_config)

        # Initialize the Agent Base class
        super().__init__(**env_config)

        # Uncomment this to find backprop errors
        # torch.autograd.set_detect_anomaly(True)

        # Initialize the rnn model
        self.rnn_model = RnnSeparator(self.rnn_config).to(self.device)
        self.rnn_model_stable = RnnSeparator(self.rnn_config).to(self.device)

        # Load pretrained model
        if pretrained:
            print('\nLoading pretrained model...\n')
            model_dict = torch.load(constants.PRETRAIN_PATH)
            self.rnn_model.rnn_model.load_state_dict(model_dict)

        # Initialize dataset related parameters
        self.bs = dataset_config['batch_size']
        self.num_updates = dataset_config['num_updates']
        self.dynamic_dataset = RLDataset(buffer=self.dataset, sample_size=self.bs)
        self.dataloader = torch.utils.data.DataLoader(self.dynamic_dataset, batch_size=self.bs)

        # Initialize network layers for DQN network
        filter_length = (stft_config['num_filters'] // 2 + 1) * 2 if stft_config is not None else 514
        total_actions = self.env.action_space.n
        network_params = {'filter_length': filter_length, 'total_actions': total_actions, 'stft_diff': self.stft_diff}
        self.q_net = DQN(network_params).to(self.device)
        self.q_net_stable = DQN(network_params).to(self.device)  # Fixed Q net

        # Tell optimizer which parameters to learn
        if pretrained:
            # Freezing the rnn model weights, only optimizing Q-net
            params = self.q_net.parameters()
        else:
            params = list(self.rnn_model.parameters()) + list(self.q_net.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)

        # Folder path where the model will be saved
        self.SAVE_PATH = dataset_config['save_path']
        self.grad_norms = None
        self.percentile = autoclip_percentile

    def update(self):
        """
        Runs the main training pipeline. Sends mix to RNN separator, then to the DQN. 
        Calculates the q-values and the expected q-values, comparing them to get the loss and then
        computes the gradient w.r.t to the entire differentiable pipeline.
        """
        # Run the update only if samples >= batch_size
        if len(self.dataset.items) < self.bs:
            return

        for index, data in enumerate(self.dataloader):
            if index > self.num_updates:
                break

            # Get the total number of time steps
            total_time_steps = data['mix_audio_prev_state'].shape[-1]

            # Reshape the mixture to pass through the separation model (Convert dual channels into one)
            # Also, rename the state to work on to mix_audio so that it can pass through remaining nussl architecture
            data['mix_audio'] = data['mix_audio_prev_state'].float().view(
                -1, 1, total_time_steps).to(self.device)
            data['action'] = data['action'].to(self.device)
            data['reward'] = data['reward'].to(self.device)
            agent_info = data['agent_info'].to(self.device)

            # Get the separated sources by running through RNN separation model
            output = self.rnn_model(data)
            output['mix_audio'] = data['mix_audio']

            # Pass then through the DQN model to get q-values
            q_values = self.q_net(output, agent_info, total_time_steps)
            q_values = q_values.gather(1, data['action'])

            with torch.no_grad():
                # Now, get q-values for the next-state
                # Get the total number of time steps
                total_time_steps = data['mix_audio_new_state'].shape[-1]

                # Reshape the mixture to pass through the separation model (Convert dual channels into one)
                data['mix_audio'] = data['mix_audio_new_state'].float().view(
                    -1, 1, total_time_steps).to(self.device)
                stable_output = self.rnn_model_stable(data)
                stable_output['mix_audio'] = data['mix_audio']
                q_values_next = self.q_net_stable(stable_output, agent_info, total_time_steps).max(1)[0].unsqueeze(-1)

            expected_q_values = data['reward'] + self.gamma * q_values_next

            # Calculate loss
            loss = F.l1_loss(q_values, expected_q_values)
            self.losses.append(loss)
            self.writer.add_scalar('Loss/train', loss, len(self.losses))

            # Optimize the model with backprop
            self.optimizer.zero_grad()
            loss.backward()

            # Applying AutoClip
            self.grad_norms = utils.autoclip(self.rnn_model, self.percentile, self.grad_norms)

            # Stepping optimizer
            self.optimizer.step()

    def choose_action(self):
        """
        Runs a forward pass though the RNN separator and then the Q-network. An action is choosen 
        by taking the argmax of the output vector of the network, where the output is a 
        probability distribution over the action space (via softmax).

        Returns:
            action (int): the argmax of the q-values vector 
        """
        with torch.no_grad():
            # Get the latest state from the buffer
            data = self.dataset[self.dataset.last_ptr]

            # Perform the forward pass (RNN separator => DQN)
            total_time_steps = data['mix_audio_new_state'].shape[-1]
            data['mix_audio'] = data['mix_audio_new_state'].float().view(
                -1, 1, total_time_steps).to(self.device)
            output = self.rnn_model(data)
            #output['mix_audio'] = data['mix_audio']
            agent_info = data['agent_info'].to(self.device)

            # action = argmax(q-values)
            q_values = self.q_net(output, agent_info, total_time_steps)
            action = q_values.max(1)[1].unsqueeze(-1)
            action = action[0].item()

            action = int(action)

            return action

    def update_stable_networks(self):
        self.rnn_model_stable.load_state_dict(self.rnn_model.state_dict())
        self.q_net_stable.load_state_dict(self.q_net.state_dict())

    def save_model(self, name):
        """
        Args:
            name (str): Name contains the episode information (To give saved models unique names)
        """
        # Save the parameters for rnn model and q net separately
        metadata = {
            'sample_rate': 8000
        }
        self.rnn_model.rnn_model.save(os.path.join(self.SAVE_PATH, 'sp_' + name), metadata)
        torch.save(self.rnn_model.state_dict(), os.path.join(self.SAVE_PATH, 'rnn_' + name))
        torch.save(self.q_net.state_dict(), os.path.join(self.SAVE_PATH, 'qnet_' + name))


class RnnSeparator(nn.Module):
    def __init__(self, rnn_config, verbose=False):
        super(RnnSeparator, self).__init__()
        self.rnn_model = nussl.ml.SeparationModel(rnn_config, verbose=verbose)

    def forward(self, x):
        return self.rnn_model(x)


class DQN(nn.Module):
    def __init__(self, network_params):
        """
        The main DQN class, which takes the output of the RNN separator and input and
        returns q-values (prob dist of the action space)

        Args:
            network_params (dict): Dict of network parameters

        Returns:
            q_values (torch.Tensor): q-values 
        """
        super(DQN, self).__init__()

        # filter_length = 258
        self.stft_diff = network_params['stft_diff']
        self.fc1 = nn.Linear(network_params['filter_length'] * 2, 64)
        self.fc2 = nn.Linear(64, network_params['total_actions'])
        self.bn = nn.BatchNorm1d(network_params['filter_length'])
        self.prelu = nn.PReLU()

    def forward(self, output, agent_info, total_time_steps):
        # Reshape the output again to get dual channels
        print('audio:', output['audio'].shape)
        output['audio'] = output['audio'].view(-1, 2, total_time_steps, 2)
        print('audio:', output['audio'].shape)
        # Perform short time fourier transform of this output
        # with torch.no_grad():
        #     output['audio'][output['audio'].abs() < 1e-4] = 1e-4
        stft_data = self.stft_diff(output['audio'], direction='transform')

        # Get the IPD and ILD features from the stft data
        ipd, ild, vol = audio_processing.ipd_ild_features(stft_data)
        # torch.Size([1, 501, 258, 2]) torch.Size([1, 501, 129, 2]) torch.Size([1, 501, 129, 2])
        print('ipd, ild, vol:', ipd.shape, ild.shape, vol.shape)

        # Create feature matrix
        X = torch.cat((ipd, ild), dim=2)
        print('X:', X.shape)  # X: torch.Size([1, 501, 387, 2])
        # agent_info = agent_info.view(-1, 3)
        # X = torch.cat((X, agent_info), dim=1)

        # Sum over the time frame axis
        X = torch.mean(X, dim=1)
        print('X:', X.shape)  # X: torch.Size([1, 387, 2])
        # with torch.no_grad():
        #     X[X.abs() < 1e-4] = 1e-4
        X = self.bn(X)

        # Flatten the features
        num_features = self.flatten_features(X)
        print('num_features:', num_features)
        X = X.view(-1, num_features)  # Shape = [Batch_size, filter_length*2]
        print('X:', X.shape)

        # Forward network pass
        X = self.prelu(self.fc1(X))
        X = self.fc2(X)
        q_values = F.softmax(X, dim=1)
        
        return q_values

    def flatten_features(self, x):
        # Flatten all dimensions except the batch
        size = x.size()[1:]
        num_features = 1
        for dimension in size:
            num_features *= dimension

        return num_features








