import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
from pyroomacoustics import MicrophoneArray, ShoeBox


def init_env(direct_sound, agent_loc, sound_loc, room_config):
    '''

    :param direct_sound: The path to audio file
    :param agent_loc: Agent/microphone location (numpy array)
    :param sound_loc: Location of audio source (list)
    :param room_config: List of room dimensions, 2d or 3d
    :return: Convolved sound
    '''

    # Load the audio file
    rate, audio = wavfile.read(direct_sound)
    print(audio.shape)

    # As the test signal is recorded on stereo microphone, it is 2d
    # Choose one of the sides
    audio = audio[:, 0]

    # Create a room using shoe box function
    room = ShoeBox(room_config)
    room.add_source(sound_loc, signal=audio)

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
    wavfile.write('../sounds/convolved_sounds/convolved_sound.wav', rate, data=scaled)

    # plot signal at microphone 0
    plt.plot(room.mic_array.signals[0, :])
    plt.show()

    return room.mic_array.signals[0:, ]


if __name__ == '__main__':
    path = '../sounds/demoaudio.wav'
    room_config = [10, 10]
    source_loc = [5, 5]
    agent_loc = np.array([2, 3]).reshape(-1, 1)

    init_env(path, agent_loc, source_loc, room_config)

