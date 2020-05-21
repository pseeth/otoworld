# NOTE: to run, need to cd into tests/, then run pytest
import sys
sys.path.append("../src")

import gym
import numpy as np

import audio_room
import utils
import room_types

def test_audio_env():
    # paths of audio files
    paths = utils.choose_random_files()

    # Shoebox Room
    room = room_types.ShoeBox(x_length=10, y_length=10)

    agent_loc = np.array([3, 8])

    # Set up the audio/gym environment
    env = gym.make(
        "audio-room-v0",
        room_config=room.generate(),
        agent_loc=agent_loc,
        corners=room.corners,
        max_order=10,
        step_size=1.0,
        direct_sources=paths,
        acceptable_radius=0.5,
    )

    # add sources
    env.add_sources()
    init_room = env.room

    # basic things of initial env
    assert(len(env.direct_sources) == len(paths))

    # test step (taking actions)
    # remember: 0,0 is at the bottom left
    env.step(action=0)  # step left
    assert(np.allclose(env.agent_loc, np.array([2, 8])))
    env.step(action=1)  # step right
    assert(np.allclose(env.agent_loc, np.array([3, 8])))
    env.step(action=2)  # step up
    assert(np.allclose(env.agent_loc, np.array([3, 9])))
    env.step(action=3)  # step down
    assert(np.allclose(env.agent_loc, np.array([3, 8])))

    # test move function
    env._move_agent([5, 5])
    assert(env.agent_loc == [5, 5])

    # ensure the room is the same dimensions
    # even though its a different q object
    new_room = env.room
    for idx, wall in enumerate(init_room.walls):
        assert(np.allclose(wall.corners, new_room.walls[idx].corners))

    # test reset


