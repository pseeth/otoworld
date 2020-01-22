import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
from pyroomacoustics import MicrophoneArray, ShoeBox


def init_env(direct_sound, agent_loc, sound_loc, room_config):
    '''

    :param direct_sound: The path(s) to audio file(s) (list)
    :param agent_loc: Agent/microphone location (numpy array)
    :param sound_loc: Location of audio source(s) (2d list, list of [x,y] locations)
    :param room_config: List of room dimensions, 2d or 3d
    :return: Convolved sound
    '''

    # Create a room using shoe box function
    room = ShoeBox(room_config)

    # Load the audio file(s), store data in parallel arrays
    audio = []
    rates = []
    for idx, audio_file in enumerate(direct_sound):
        r, a = wavfile.read(audio_file)
        print("Rate: ", r)

        # store
        rates.append(r)

        # If sound is recorded on stereo microphone, it is 2d
        # Choose one of the sides
        if len(a.shape) > 1:
            a = a[:, 0]

        audio.append(a)

        room.add_source(sound_loc[idx], signal=a)

    # Create the microphone array
    mic = MicrophoneArray(agent_loc, room.fs)
    room.add_microphone_array(mic)

    # Impulse responses
    room.compute_rir()
    # plt.plot(room.rir[0][0])
    # plt.show()

    # Convolve the sounds
    room.simulate()
    # print(room.mic_array.signals.shape)
    data = room.mic_array.signals[0, :]

    # Data needs to be in integer before being saved
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)

    # Save the convolved sound
    # NOTE: rates[0] may not be ideal, may need to make more robust later
    wavfile.write('../sounds/convolved_sounds/convolved_sound.wav', rates[0], data=scaled)

    # plot signal at microphone 0
    plt.plot(room.mic_array.signals[0, :])
    plt.title('Convolution of Signals and Room Impulse Response')
    plt.show()

    return room.mic_array.signals[0:, ]


if __name__ == '__main__':
    # paths of audio files
    paths = ['../sounds/dry_recordings/dev/050_subset/050a050a.wav', '../sounds/dry_recordings/dev/051_subset/051a050b.wav']

    # dimensions of room
    room_config = [10, 10]

    # locations of audio sources (direct correspondence with paths list, e.g. [5,5] is the location of 050a050a.wav)
    source_loc = [[5, 5], [7, 8]]

    # location of agent (or microphone in this case)
    agent_loc = np.array([2, 3]).reshape(-1, 1)

    init_env(paths, agent_loc, source_loc, room_config)



