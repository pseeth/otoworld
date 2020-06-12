import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random
import torch

import constants

def autoclip(model, percentile, grad_norms=None):
    if grad_norms is None:
        grad_norms = []
    
    def _get_grad_norm(model):
        total_norm = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    grad_norms.append(_get_grad_norm(model))
    clip_value = np.percentile(grad_norms, percentile)

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), clip_value)
    return grad_norms


def choose_random_files(source_folders_dict):
    """
    Function returns random source files from provided folders.
    
    Args:
        source_folders_dict (Dict[str, int]): specify how many source files to choose from each folder
            e.g.
                {
                    'car_horn_source_folder': 1,
                    'phone_ringing_source_folder': 1
                }
            
            This would choose 1 source file from each folder
    Returns:
        paths (List[str]): the paths to two wav files
    """
    paths = []

    for folder, num_sources in source_folders_dict.items():
        files = os.listdir(folder)

        source_files = []
        random_indices = np.random.permutation(len(files))
        for i in random_indices:
            if files[i].endswith(constants.AUDIO_EXTENSION):
                source_files.append(os.path.join(folder, files[i]))

            if len(source_files) == num_sources:
                break
        
        paths.extend(source_files)
            
    print('SOURCE FILE PATHS:', paths)
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


def log_reward_vs_steps(rewards_per_episode):
    """
    This function logs the rewards per episode in order to plot the rewards vs. step for each episode. 
    The lists are stored in pickle files. The pairs (dist, steps) are in parallel lists, indexed by the 
    episode number.

    Args:
        rewards_per_episode (List[float]): rewards gained per episode
    """
    # create data folder
    if not os.path.exists(constants.DATA_PATH):
        os.makedirs(constants.DATA_PATH)

    # write objects
    pickle.dump(
        rewards_per_episode, open(os.path.join(
            constants.DATA_PATH, constants.REWARD_URL), "wb"),
    )


def plot_reward_vs_steps():
    """
    Plots the reward vs step for an episode.
    """
    with open(os.path.join(constants.DATA_PATH, constants.REWARD_URL), "rb") as f:
        rewards = pickle.load(f)

    reward = rewards[0]
    plt.scatter(list(range(len(reward))), reward)
    plt.title("Reward vs. Number of Steps")
    plt.xlabel("Step")
    plt.ylabel("Reward")

    plt.show()

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
    """Empty and re-create the buffer data folders"""
    if os.path.exists(constants.DIR_PREV_STATES):
        shutil.rmtree(constants.DIR_PREV_STATES)
    os.makedirs(constants.DIR_PREV_STATES)
    if os.path.exists(constants.DIR_NEW_STATES):
        shutil.rmtree(constants.DIR_NEW_STATES)
    os.makedirs(constants.DIR_NEW_STATES)
    if os.path.exists(constants.DIR_DATASET_ITEMS):
        shutil.rmtree(constants.DIR_DATASET_ITEMS)
    os.makedirs(constants.DIR_DATASET_ITEMS)


def clear_models_folder(save_path):
    """Empty/clear and re-create save_path folder to save models.
    
    Args:
        save_path (str): path to folder where models will be stored
    """
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)


