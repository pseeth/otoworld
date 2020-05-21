# Reinforcement learning for computer audition

Authors: Grant Gasser, David Terpay, Omkar Ranadive
Advisor: Prem Seetharaman

Applying reinforcement learning to perform source separation.

### Requirements:

#### Environment
We recommend using a conda environment for this project:
* `conda create -n myenv python=3.7`
* Then activate the environment: `conda activate myenv`

#### Installing the Pyroomacoustics Package
Since we use a newer version than what is available through traditional install methods, it is necessary
to to clone the Pyroomacoustics library from the `master` branch on github and use that version. 
* Clone pyroomacoustics like so:

    `git clone https://github.com/LCAV/pyroomacoustics.git`
* With your environment activated, run `pip install -e .` 
* You should see "Successfully installed pyroomacoustics"
* The reason for this is that there's a parameter in the old version that is not large enough, causing the 
`Room.is_inside` function to fail. With the version on the master branch, there is a new constant in `parameters.py` 
named `room_isinside_max_iter` that is set to 20 (previously 5, too low). For training, we set this to 
`room_isinside_max_iter = 50` to be even more cautious. 

#### Using Poetry (For us)
- For for package and dependency management, we use Poetry
- [How to install Poetry](https://python-poetry.org/docs/#installation)
- Run `poetry install` to install dependencies for the project (listed in `pyproject.toml`)
- Run `poetry export -f requirements.txt` to create/update requirements based on `pyproject.toml`

#### Otherwise
Assuming `requirements.txt` is up do date and you've successfully installed 
pyroomacoustics as directed above, run `pip install -r requirements.txt`

### Run
CD into `experiments/` and run a file like so: `python experiment1.py`

### Testing
CD into `tests/` and run `pytest` 

### Formatting
`pip install black`

`black {source_file_or_directory}`

To set the max line length to 99: `black -l 99 {source_file_or_directory}`

### Model?
- Old way: mask in (0, 1); element-wise multiply and get the original source
- Train anchor points and "red" points 
- Steps:
    1. Project L and R channels to separate embedding spaces (features)
    2. Create anchors
    3. Track size (loudness) of clusters/sources in matrix (embedding spaces); go to louder side (L or R)
    4. Element-wise multiple anchors embedding to get mask
    5. Mask * respective channel, then sum that
    4. Embedding -> 
        - loudness of source0 in L ear
        - loudness of source1 in R ear
        - loudness of source0 in L ear
        - loudness of source1 in R ear
    5. Take these with linear layer and map to action space 

### To Do (Winter Quarter)
- [X] Split up Pyroom initiliazation and convolution calculation (in `basic_room.py`)
- [X] If stereo file, take mean of 2 channels (in `basic_room.py`)
- [X] Replace `wavfile.read` with `librosa` (add with poetry)
- [X] Figure out how to poetry `export` to a `requirements.txt` so users don't have to use `poetry`
- [X] Need 1 more microphone (2 total)
- [X] Randomize sound source files
- [X] Put sources randomly in the environment 
- [X] Place the agent roughly equidistant to the sources 
- [X] Keep the distance between mics 20 cm 
- [X] Add the orientation actions (left x degrees, right x degrees) to set of actions 
    - (NOTE: angle is determined in radians using the `linear_2D_array` function, use `np.pi`) 
- [X] More rooms (randomly generated if possible), simple rooms like hexagon, octagon 
- [X] Cut the lengths of the sources based on length of shortest source.
- [X] Make step size tunable
- [X] Updated movements to be able to deal with floats
- [X] Get running on gpubox
- [X] Refactor loop in `audio_env.py` in `add_sources` function to support turning on and off sources
- [X] CLEAN UP: remove unecessary print statements (commented out), functions, classes, and files no longer used
- [X] Run 1000 episodes and plot initial distances to src and number of steps to reach target (see `steps_and_dist.png`)
- [X] One Action structure: U, D, L, R, rotate left, rotate right (similar to current)
- [X] Have agent turn off both sources (move randomly in small environment)
- [X] Measure throughput (how many steps we can run per second without plotting with random agent)
  - ~ **20 steps/second**
- [X] Store in buffer (S, A, S', R) which is (prev audio, action, current audio, reward), refer to [DQN code](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [X] Clean up: remove DSstores from git, remove `temp.py`, remove classes from `rl_agent.py` except for random agent and oroom perfect agent, run black to fix formatting
- [X] Set up experiment structure: `exp/exp1.py` and `src/env`, `src/agent`, `src/room`, `src/utils` (store_data), `src/models`, `src/constants.py` and start adding `tests/`
- [X] Reward: -.1 for each step, 10 for turning off source (put in `constants`)
- [X] Make # of sources a parameter
- [X] Put `is_inside` in try/catch statement in case it fails
- [X] Remove notion of targets
    - Loop thru all srcs and check if agent is close enough to turn off
- [X] REMEMBER to clone PRA newest version on gpubox before training

### To Do (Spring Quarter)
- [X] Using `AudioSignal` objects to store data
    - Similar to [these mixing functions](https://github.com/interactiveaudiolab/nussl/blob/refactor/nussl/core/mixing.py)

- [X] Store observations from environment into dataset subclass (`BaseDataset`) 
- [X] Make all sources same length - already done!
- [X] Make `new_state` when agent finds 2nd source (when `reward=10`) silence (i.e. `np.zeros` audioSignal)
- [X] Ensure agent doesn't spawn too close (or on top of) a source (sometimes it spawns too close and turns it off on 0th step)
- [X] Refactor `agent.py` like so: One `Agent` base class with `fit` function
    - Subclasses (Random Agent, Model Agent) implement different `choose_action` functions (based on current lines 137,138)
- [X] Add option to choose new/different sources after resetting env for each episode
- [X] Plot of step vs. reward within episode    
- [X] Limit # of buffer items (json files/prev wav files/new wav files) using `MAX_BUFFER_ITEMS`
    - Add push/pop functions to `BufferData` class to limit size of `items` list using
    - Be sure that the # of files written into each folder (`data/dataset_items/`, `data/new_states`, `data/prev_states`) is also < `MAX_BUFFER_ITEMS`     
- [X] Implement [nussl transforms](https://nussl.github.io/docs/tutorials/datasets.html#Transforms) to prepare for training, transforms passed to `BufferData __init__`
    - [getExcerpt](https://nussl.github.io/docs/datasets.html#nussl.datasets.transforms.GetExcerpt): samples random 
    frames from spectrograms
    - getAudio transform: takes audio data and returns 
    - [ToSeparationModel](https://nussl.github.io/docs/datasets.html#nussl.datasets.transforms.ToSeparationModel)
- [X] Use [PyTorch data loader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- [X] Add update function to AgentBase class (random agent its just a pass, Q-learning update for other model)
- [X] Add print to logging 
- [X] Using [this sampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler), ensure we
are sampling the same amount of items in `items` from each episode (i.e. if `batch_size=25` and we have 5 episodes, then
choosing 5 samples from each episode, regardless of the size of the episode)
- [X] Run on gpubox
- [In progress] Write smoke/unit tests
- [ ] Output of separation model `[25, 2, 32000, 2]` => [`ipd_ild_features()`](https://github.com/nussl/nussl/blob/551d5e46c23dadea328e0473e3038d99cd0c1ce6/nussl/core/audio_signal.py#L1096) => sum ipd and ild across all time steps for each spectogram
    - What would be the output dim of that function? (for just one spectrogram, the result would be `[f, 1, 2]` like `[freq, time bin, ipd and ild]`)
    - [Video](https://northwestern.zoom.us/rec/play/u5QucuH8-zs3Ht2Q5ASDBqQvW465KK2shyFK__QJnRy1UnMGY1qlNecQY7HpTSf8zvbjcqTAP0wpqMuX?continueMode=true)
    - Then reshape [f * ipd | f * ild] (one spectrogram's ipd/ild after being summed across time) and 
    map that to 6-dim action space w/ softmax using `torch.nn.Linear(2f, 6)`
    - nussl [`SpatialClustering`](https://github.com/nussl/nussl/blob/master/nussl/separation/spatial/spatial_clustering.py) class 
    - nussl [STFT](https://github.com/nussl/nussl/blob/master/nussl/ml/networks/modules/filter_bank.py#L240)

### Timeline
* Realistic: June 15 [ICML Workshop](https://icml-sas.gitlab.io/)
* July, some conference
* Maybe paper for ICML or ICLR 

### Resources: 
#### Environments
* [PyRoom Acoustics](https://github.com/LCAV/pyroomacoustics)
* [Gym mini world](https://github.com/maximecb/gym-miniworld)
* [Gym mini grid](https://github.com/maximecb/gym-minigrid)
* [Pytorch DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* [Pyorch Ignite](https://pytorch.org/ignite/)
