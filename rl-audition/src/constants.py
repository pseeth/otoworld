import os

# source data
DIR_MALE = "../sounds/dry_recordings/dev/051/"
DIR_FEMALE = "../sounds/dry_recordings/dev/050/"
DIR_CAR = '../sounds/car/'
DIR_PHONE = '../sounds/siren/'
AUDIO_EXTENSION = ".wav"

# saved data during experiment
DATA_PATH = "../data"
DIR_PREV_STATES = os.path.join(DATA_PATH, 'prev_states/')
DIR_NEW_STATES = os.path.join(DATA_PATH, 'new_states/')
DIR_DATASET_ITEMS = os.path.join(DATA_PATH, 'dataset_items/')
MODEL_SAVE_PATH = '../models/'
DIST_URL = "init_dist_to_target.p"
STEPS_URL = "steps_to_completion.p"
REWARD_URL = "rewards_per_episode.p"

# audio stuff
RESAMPLE_RATE = 8000

# env stuff
DIST_BTWN_EARS = 0.15

# max and min values of exploration rate
MAX_EPSILON = 0.9
MIN_EPSILON = 0.01

# reward structure
STEP_PENALTY = -0.5
TURN_OFF_REWARD = 100.0  # keep this a float, otherwise Pytorch dataloader will throw errors
ORIENT_PENALTY = -0.1 

# dataset
MAX_BUFFER_ITEMS = 10000
