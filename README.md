# Reinforcement learning for computer audition

Authors: Grant Gasser, David Terpay, Omkar Ranadive, Prem Seetharaman

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
- Week of 2/10
- Week of 2/17
- Week of 2/24
- Week of 3/02
- Week of 3/09

### Requirements:
- Create a conda environment: `conda create -n [your-env-name] python=3.7`
- Install [Poetry](https://python-poetry.org/docs/#installation) - we use Poetry for package and dependency management
- Run `poetry install` to install dependencies for the project (listed in `pyproject.toml`)

### Run (Recommended)
`poetry run python basic_room2.py `

### Tentative Plan
- [X] Start with `f(direct_sound, agent_loc, sound_loc, room_config) => convolved_sound` using Pyroom
- [X] Extend to a RL environment (Gym)

### Model
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

### Directions
- Keep < 8K Hz sample rate (8,000 samples/time intervals per second)

### To-do (High Level)
- [ ] Split up Pyroom initiliazation and convolution calculation (in `basic_room.py`)
- [X] If stereo file, take mean of 2 channels (in `basic_room.py`)
- [X] Replace `wavfile.read` with `librosa` (add with poetry)
- [X] Figure out how to poetry `export` to a `requirements.txt` so users don't have to use `poetry`
- [ ] Need 1 more microphone (2 total)
- [ ] Orient mic array in different directions (for rotation) (in `basic_room.py`)
- [ ] Randomize sound source files
- [ ] Extend configuration (`room_config`) to make different rooms (multiple shoeboxes)
    - Each create 3 (unique) rooms and have oracle agent go to sources

### Ideas for Games
* Agent should find source and "turn it off" (agent reaches same grid location)
    - Maybe record impulse responses to see change as agent gets closer
    - Reward structure: continuous reward based on how loud the environment is, reward for turning off source
    - Action space: rotate_left (x degrees), rotate_right (x degrees); then step (however far)
    - State space: tbd

### Resources: 
#### Environments
* Gym mini world: https://github.com/maximecb/gym-miniworld
* Gym mini grid: https://github.com/maximecb/gym-minigrid
