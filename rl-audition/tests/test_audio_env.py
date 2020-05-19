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

    # Uncomment for Polygon Room
    # room = room_types.Polygon(n=6, r=2, x_center=5, y_center=5)

    agent_loc = np.array([3, 8])

    # Set up the gym environment
    env = gym.make(
        "audio-room-v0",
        room_config=room.generate(),
        agent_loc=agent_loc,
        corners=room.corners,
        max_order=10,
        step_size=1.0,
        direct_sources=paths,
        acceptable_radius=0.8,
    )

    # add sources
    env.add_sources()
    init_room = env.room

    # basic things of initial env
    assert(len(env.direct_sources) == len(paths))

    # test step (taking actions)
    # hard to test, may not take step because of boundaries
    # remember: 0,0 is at the bottom left
    env.step(action=0)  # step left
    assert(np.allclose(env.agent_loc, np.array([2, 8])))
    env.step(action=1)  # step right
    assert(np.allclose(env.agent_loc, np.array([3, 8])))
    env.step(action=2)  # step up
    assert(np.allclose(env.agent_loc, np.array([3, 9])))
    env.step(action=3)  # step down
    assert(np.allclose(env.agent_loc, np.array([3, 8])))

    # resetting the env (after removing a source or starting a new episode)
    env.reset(removing_source=0)
    assert(len(env.direct_sources) < len(paths))

    # ensure the room is the same even with a different object
    new_room = env.room
    for idx, wall in enumerate(init_room.walls):
        assert(np.allclose(wall.corners, new_room.walls[idx].corners))





