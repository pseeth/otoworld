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
  - Deliverable: Want environment setup and be onbservable; `file = env.render()`, save audio file with librosa
- Week of 1/27
- Week of 2/03
- Week of 2/10
- Week of 2/17
- Week of 2/24
- Week of 3/02
- Week of 3/09

### Requirements:
* Install [Poetry](https://python-poetry.org/docs/#installation) - we use Poetry for package and dependency management
* Run `poetry install` to install dependencies for the project (listed in `pyproject.toml`)

### Tentative Plan
- [X] Start with `f(direct_sound, agent_loc, sound_loc, room_config) => convolved_sound` using Pyroom
- [ ] Extend to a RL environment (Gym)

## Directions
- Keep < 8K Hz sample rate (8,000 samples/time intervals per second)

### To-do (High Level)
- [ ] Split up Pyroom initiliazation and convolution calculation (in `basic_room.py`)
- [ ] If stereo file, take mean of 2 channels (in `basic_room.py`)
- [ ] Orient mic array in different directions (for rotation) (in `basic_room.py`)
- [ ] Replace `wavfile.read` with `librosa` (add with poetry)
- [ ] Figure out how to poetry `export` to a `requirements.txt` so users don't have to use `poetry`

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
