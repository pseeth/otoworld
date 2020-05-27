import nussl
import json
import os
import constants
import numpy as np
from torch.utils.data import IterableDataset


class BufferData(nussl.datasets.BaseDataset):
    def __init__(self, folder, to_disk=False, transform=None):
        """

        Args:
            folder (string): File path to store the data. Put any string when not saving to disk.
            to_disk (bool): When true, buffer will be saved to disk, else data will be stored directly in memory.
            transform (transforms.* object): A transforms to apply to the output of
              ``self.process_item``. If using transforms.Compose, each transform will be
              applied in sequence. Defaults to None.
        """
        # Circular buffer parameters
        self.MAX_BUFFER_ITEMS = constants.MAX_BUFFER_ITEMS
        self.ptr = 0
        self.items = []
        self.metadata = {}
        self.full_buffer = False
        self.to_disk = to_disk

        # Make sure the relevant directories exist
        if not os.path.exists(constants.DIR_PREV_STATES):
            os.mkdir(constants.DIR_PREV_STATES)
        if not os.path.exists(constants.DIR_NEW_STATES):
            os.mkdir(constants.DIR_NEW_STATES)
        if not os.path.exists(constants.DIR_DATASET_ITEMS):
            os.mkdir(constants.DIR_DATASET_ITEMS)

        super().__init__(folder=folder, transform=transform)

    def get_items(self, folder):
        """
        Superclass: "This function must be implemented by whatever class inherits BaseDataset.
        It should return a list of items in the given folder, each of which is
        processed by process_items in some way to produce mixes, sources, class
        labels, etc."

        Implementation: Adds file paths to items list. Keeps list under MAX_ITEMS.

        Args:
            folder (str): location that should be processed to produce the list of files
        Returns:
            list: list of items (path to json files in our case) that should be processed
        """
        # We are initializing the dataset without having any actual data, just return self.items
        return self.items

    def process_item(self, item):
        """
        Superclass: Each file returned by get_items is processed by this function. For example,
        if each file is a json file containing the paths to the mixture and sources,
        then this function should parse the json file and load the mixture and sources
        and return them.
        Exact behavior of this functionality is determined by implementation by subclass."

        Implementation: read json of format:
            {'prev_state': '../data/prev_states/prev8-224.wav',
             'action': 0,
             'reward': -0.1,
             'new_state': '../data/new_states/new8-224.wav'}

        convert the wav files to AudioSignals and return and output dict:
            {
              'observations': {
                 'prev_state': AudioSignal,
                 'new_state': AudioSignal,
              }
              'reward': -0.1
              'action': 0
            }

        Args:
            item (object): the item that will be processed by this function. Input depends
              on implementation of ``self.get_items``.
        Returns:
            This should return a dictionary that gets processed by the transforms.
        """
        if self.to_disk:
            cur_path = os.path.join(constants.DIR_DATASET_ITEMS, str(item) + '.json')

            with open(os.path.join(cur_path), 'r') as json_file:
                output = json.load(json_file)

            # convert wav files to AudioSignal objects
            prev_state = nussl.AudioSignal(output['prev_state'])
            new_state = nussl.AudioSignal(output['new_state'])

        else:
            # If we are directly getting the item from memory, it will be a dictionary
            output = item.copy()
            # print("Ouput: ", output.keys())
            prev_state, new_state = output['prev_state'], output['new_state']

        # convert to output dict format
        del output['prev_state'], output['new_state']
        output['prev_state'], output['new_state'] = prev_state, new_state
        output['reward'] = np.array([output['reward']], dtype='float32')
        output['action'] = np.array([output['action']])
        return output

    def append(self, item):
        """
        Override the default append function to work as circular buffer
        Args:
            item (object): Item to append to the list

        Returns: Nothing (Item is appended to the circular buffer in place)
        """
        if self.full_buffer:
            self.items[self.ptr] = item
            self.ptr = (self.ptr + 1) % self.MAX_BUFFER_ITEMS
        else:
            self.items.append(item)
            if len(self.items) == self.MAX_BUFFER_ITEMS:
                self.ptr = 0
                self.full_buffer = True

    def write_buffer_data(self, prev_state, action, reward, new_state, episode, step):
        """
        Writes states (AudioSignal objects) to .wav files and stores this buffer data
        in json files with the states keys pointing to the .wav files. The json files
        are to be read by nussl.datasets.BaseDataset subclass as items.

        E.g. {
            'prev_state': '/path/to/previous/mix.wav',
            'reward': [the reward obtained for reaching current state],
            'action': [the action taken to reach current state from previous state]
            'current_state': '/path/to/current/mix.wav',
        }

        The unique file names are structured as path/[prev or new]-[episode #]-[step #]

        Args:
            prev_state (nussl.AudioSignal): previous state to be converted and saved as .wav file
            action (int): action
            reward (int): reward
            new_state (nussl.AudioSignal): new state to be converted and saved as wav file
            episode (int): which episode we're on, used to create unique file name for state
            step (int): which step we're on within episode, used to create unique file name for state
        """
        if episode not in self.metadata:
            self.metadata[episode] = 1
        else:
            self.metadata[episode] += 1

        if not self.to_disk:
            buffer_dict = {
                'prev_state': prev_state,
                'action': action,
                'reward': reward,
                'new_state': new_state
            }
            self.append(buffer_dict)
        else:
            # Unique file names for each state
            cur_file = str(episode) + '-' + str(step)
            prev_state_file_path = os.path.join(
                constants.DIR_PREV_STATES, 'prev' + cur_file + '.wav'
            )
            new_state_file_path = os.path.join(
                constants.DIR_NEW_STATES, 'new' + cur_file + '.wav'
            )
            dataset_json_file_path = os.path.join(
                constants.DIR_DATASET_ITEMS, cur_file + '.json'
            )

            prev_state.write_audio_to_file(prev_state_file_path)
            new_state.write_audio_to_file(new_state_file_path)

            # write to json
            buffer_dict = {
                'prev_state': prev_state_file_path,
                'action': action,
                'reward': reward,
                'new_state': new_state_file_path
            }

            # If buffer is full, delete old files from disk
            if self.full_buffer:

                file_to_delete = self.items[self.ptr]
                os.remove(os.path.join(constants.DIR_PREV_STATES, 'prev' + file_to_delete + '.wav'))
                os.remove(os.path.join(constants.DIR_NEW_STATES, 'new' + file_to_delete + '.wav'))
                os.remove(os.path.join(constants.DIR_DATASET_ITEMS, file_to_delete + '.json'))

            with open(dataset_json_file_path, 'w') as json_file:
                json.dump(buffer_dict, json_file)

                # KEY PART: append to items list of dataset object (our buffer)
                self.append(cur_file)


class RLDataset(IterableDataset):
    """
    Dataset which gets updated as buffer gets filled
    """
    def __init__(self, buffer, sample_size):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        for index, data in enumerate(self.buffer):
            if index > self.sample_size:
                yield data