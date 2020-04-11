import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import shutil

import constants


def choose_random_files(num_sources=2):
    """
	Function returns source random files using the directory constants. It chooses one file from the
	female recordings and one from the male recordings

	Args:
	    num_sources (int): number of sources to place in the room

	Returns:
		paths (List[str]): the paths to two wav files
	"""
    paths = []

    for i in range(num_sources):
        # randomly choose male or female voice folder
        dir = random.choice([constants.DIR_MALE, constants.DIR_FEMALE])
        files = os.listdir(dir)

        file = ""
        while constants.AUDIO_EXTENSION not in file:
            idx = np.random.randint(len(files), size=1)[0]
            file = files[idx]

        paths.append(os.path.join(dir, file))

    return paths


def log_dist_and_num_steps(init_dist_to_target, steps_to_completion):
    """
    This function logs the initial distance between agent and target source and number of steps 
    taken to reach target source. The lists are stored in pickle files. The pairs (dist, steps) are in parallel
    lists, indexed by the episode number.

    Args:
        init_dist_to_target (List[float]): initial distance between agent and target src (size is number of episodes)
        steps_to_completion (List[int]): number of steps it took for agent to get to source
    """
    # create data folder
    if not os.path.exists(constants.DATA_PATH):
        os.makedirs(constants.DATA_PATH)

    # write objects
    pickle.dump(
        init_dist_to_target, open(os.path.join(constants.DATA_PATH, constants.DIST_URL), "wb"),
    )
    pickle.dump(
        steps_to_completion, open(os.path.join(constants.DATA_PATH, constants.STEPS_URL), "wb"),
    )


def plot_dist_and_steps():
    """Plots initial distance and number of steps to reach target"""
    with open(os.path.join(constants.DATA_PATH, constants.DIST_URL), "rb") as f:
        dist = pickle.load(f)
        avg_dist = np.mean(dist)

    with open(os.path.join(constants.DATA_PATH, constants.STEPS_URL), "rb") as f:
        steps = pickle.load(f)
        avg_steps = np.mean(steps)

    plt.scatter(dist, np.log(steps))
    plt.title("Number of Steps and Initial Distance")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Log(# of Steps to Reach Target)")
    plt.text(
        5,
        600,
        "Avg Steps: " + str(int(avg_steps)),
        size=15,
        rotation=0.0,
        ha="right",
        va="top",
        bbox=dict(boxstyle="square", ec=(1.0, 0.5, 0.5), fc=(1.0, 0.8, 0.8),),
    )
    plt.text(
        5,
        500,
        "Avg Init Dist: " + str(int(avg_dist)),
        size=15,
        rotation=0.0,
        ha="right",
        va="top",
        bbox=dict(boxstyle="square", ec=(1.0, 0.5, 0.5), fc=(1.0, 0.8, 0.8),),
    )

    plt.show()


def create_buffer_data_folders():
    # empty and re-create the folders
    if os.path.exists(constants.DIR_PREV_STATES):
        shutil.rmtree(constants.DIR_PREV_STATES)
    os.makedirs(constants.DIR_PREV_STATES)
    if os.path.exists(constants.DIR_NEW_STATES):
        shutil.rmtree(constants.DIR_NEW_STATES)
    os.makedirs(constants.DIR_NEW_STATES)
    if os.path.exists(constants.DIR_DATASET_ITEMS):
        shutil.rmtree(constants.DIR_DATASET_ITEMS)
    os.makedirs(constants.DIR_DATASET_ITEMS)


def write_buffer_data(prev_state, action, reward, new_state, episode, step, dataset):
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
        dataset (subclass of nussl.datasets.AudioSignal): dataset to append item to
    """
    # unique file names for each state
    prev_state_file_path = os.path.join(
        constants.DIR_PREV_STATES, 'prev' + str(episode) + '-' + str(step) + '.wav'
    )
    new_state_file_path = os.path.join(
        constants.DIR_NEW_STATES, 'new' + str(episode) + '-' + str(step) + '.wav'
    )
    dataset_json_file_path = os.path.join(
        constants.DIR_DATASET_ITEMS, str(episode) + '-' + str(step) + '.json'
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

    with open(dataset_json_file_path, 'w') as json_file:
        json.dump(buffer_dict, json_file)

        # KEY PART: append to items list of dataset object (our buffer)
        dataset.items.append(json_file.name)
