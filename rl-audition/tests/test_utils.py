# NOTE: to run, need to cd into tests/, then run pytest
import sys
sys.path.append("../src")

import numpy as np

import utils
import constants

def test_choose_random_files():
    # test a range of different numbers of files 
    for num_sources in range(5):
        num_sources = np.random.randint(1, 5)
        paths = utils.choose_random_files(num_sources=num_sources)

        assert(len(paths) == num_sources)

        # ensure we collect from correct folder
        random_file = np.random.choice(paths)
        assert(
            random_file.startswith(constants.DIR_FEMALE) \
            or random_file.startswith(constants.DIR_MALE)
        )

        # .wav or .mp3 or.. (need to be audio files)
        assert(random_file.endswith(constants.AUDIO_EXTENSION))
