# Reinforcement learning for computer audition

Authors: Grant Gasser, David Terpay, Omkar Ranadive
Advisor: Prem Seetharaman

Applying reinforcement learning to perform source separation.

Project timeline
- Week of 1/13
  - Learn: RL, Double Deep Q-Learning (DDQN), SOTA (vision-related tasks), acoustics (how sound travels)
  - Code: Existing implementations (Gym), Pyroom acoustics
  - Deliverable: Want `f(direct_sound, agent_loc, sound_loc, room_config) => convolved_sound`
- Week of 1/20
  - RL review 
  - Deliverable: Want environment setup and be observable; `file = env.render()`, save audio file with librosa
- Week of 1/27
  - Presentation on Thursday (RL overview + Project Explanation)
  - Deliverable: See To-do
- Week of 2/03
  - See To-do
- Week of 2/10
  - See To-do
- Week of 2/17
  - See To-do
- Week of 2/24
  - See To-do
- Week of 3/02
  - See To-do
- Week of 3/09

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

#### Using Poetry
- For for package and dependency management, we use Poetry
- [How to install Poetry](https://python-poetry.org/docs/#installation)
- Run `poetry install` to install dependencies for the project (listed in `pyproject.toml`)
- Run `poetry export -f requirements.txt` to create/update requirements based on `pyproject.toml`

#### Otherwise
Assuming `requirements.txt` is up do date and you've successfully installed 
pyroomacoustics as directed above, run `pip install -r requirements.txt`

### Run
`python main.py`

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

### To Do (High Level)
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
- [ ] Add option to choose new/different sources after resetting env for each episode
- [X] REMEMBER to clone PRA newest version on gpubox before training

### To Do (Spring Quarter)
- [ ] Refactor code using `nussl`
- [X] Using `AudioSignal` objects to store data
    - Similar to [these mixing functions](https://github.com/interactiveaudiolab/nussl/blob/refactor/nussl/core/mixing.py)

- [ ] Store observations from environment into dataset subclass (`BaseDataset`) 
- [ ] Make all sources same length (i.e. 10 seconds)
- [X] Make `new_state` when agent finds 2nd source (when `reward=10`) silence (i.e. `np.zeros` audioSignal)
- [ ] Plot of step vs. reward within episode    
    
### RL Setup
* Agent should find source and "turn it off" (agent reaches same grid location)
    - State: convolved sound
    - Action space: rotate_left (x degrees), rotate_right (x degrees), step (L, R, U, D)
    - Small negative reward for each action (-0.1), large reward for turning off source (+10)
    - Store replay buffer (S, A, R, S')

### Resources: 
#### Environments
* [PyRoom Acoustics](https://github.com/LCAV/pyroomacoustics)
* [Gym mini world](https://github.com/maximecb/gym-miniworld)
* [Gym mini grid](https://github.com/maximecb/gym-minigrid)
* [Pytorch DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* [Pyorch Ignite](https://pytorch.org/ignite/)
