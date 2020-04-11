import nussl
import json
import os


class BufferData(nussl.datasets.BaseDataset):
    def get_items(self, folder):
        """
        This function must be implemented by whatever class inherits BaseDataset.
        It should return a list of items in the given folder, each of which is
        processed by process_items in some way to produce mixes, sources, class
        labels, etc.
        Args:
            folder (str): location that should be processed to produce the list of files
        Returns:
            list: list of items (json files in our case) that should be processed
        """
        items = []
        for file in os.listdir(folder):
            items.append(os.path.join(folder, file))
        return items

    def process_item(self, item):
        """Each file returned by get_items is processed by this function. For example,
        if each file is a json file containing the paths to the mixture and sources,
        then this function should parse the json file and load the mixture and sources
        and return them.
        Exact behavior of this functionality is determined by implementation by subclass.
        Args:
            item (object): the item that will be processed by this function. Input depends
              on implementation of ``self.get_items``.
        Returns:
            This should return a dictionary that gets processed by the transforms.
        """
        with open(os.path.join(item), 'r') as json_file:
            output = json.load(json_file)

        # convert wav files to AudioSignal objects
        prev_state = nussl.AudioSignal(output['prev_state'])
        new_state = nussl.AudioSignal(output['new_state'])

        output['prev_state'] = prev_state
        output['new_state'] = new_state

        return output
