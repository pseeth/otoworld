import os

# file paths and extensions
DIR_MALE = "../sounds/dry_recordings/dev/051_subset/"
DIR_FEMALE = "../sounds/dry_recordings/dev/050_subset/"
DATA_PATH = "../data"
DIR_PREV_STATES = os.path.join(DATA_PATH, 'prev_states/')
DIR_NEW_STATES = os.path.join(DATA_PATH, 'new_states/')
DIR_DATASET_ITEMS = os.path.join(DATA_PATH, 'dataset_items/')
DIST_URL = "init_dist_to_target.p"
STEPS_URL = "steps_to_completion.p"
AUDIO_EXTENSION = ".wav"

# max and min values of exploration rate
MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

# reward structure
STEP_PENALTY = -0.1
TURN_OFF_REWARD = 10

# dataset
MAX_BUFFER_ITEMS = 20000
