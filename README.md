# Reinforcement learning for computer audition

Authors: Grant Gasser, David Terpay, Omkar Ranadive, Prem Seetharaman

Applying reinforcement learning to perform source separation.

Project timeline
- Week of 1/13
  - Learn: RL, Double Leep Q-Learning, SOTA (vision-related tasks), acoustics (how sound travels)
  - Code: Existing implementations (Gym), Pyroom acoustics
  - Deliverable: Want `f(direct_sound, agent_loc, sound_loc, room_config) => convolved_sound`
- Week of 1/20
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

### Project Brainstorm
1. Start with `f(direct_sound, agent_loc, sound_loc, room_config) => convolved_sound` using Pyroom
2. Extend to a RL environment (Gym)

### Resources: 
#### Environments
* Gym mini world: https://github.com/maximecb/gym-miniworld
* Gym mini grid: https://github.com/maximecb/gym-minigrid
