import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import fftconvolve
from pyroomacoustics import MicrophoneArray, ShoeBox
#import simpleaudio as sa
# NOTE: the resample rate is defaulted to 8Khz for computational reasons
RESAMPLE_RATE = 8000


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
    for idx, audio_file in enumerate(direct_sound):
        # Audio will be automatically resampled to the given rate (default sr=8000).
        a, __ = librosa.load(audio_file, sr=RESAMPLE_RATE)

        # If sound is recorded on stereo microphone, it is 2d
        # take the mean of the two stereos
        if len(a.shape) > 1:
            a = np.mean(a, axis=0)

        audio.append(a)

        room.add_source(sound_loc[idx], signal=a)

    # Create the microphone array
    print(room.mic_array)
    mic = MicrophoneArray(agent_loc.reshape(-1, 1), room.fs)
    room.add_microphone_array(mic)

    # Impulse responses
    room.compute_rir()
    print(type(room.rir))
    # plt.plot(room.rir[0][0])
    # plt.show()

    # Convolve the sounds
    room.simulate()
    print(room.mic_array.signals.shape)
    data = room.mic_array.signals[0, :]
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)

    #play_obj = sa.play_buffer(scaled, num_channels=1, bytes_per_sample=2, sample_rate=RESAMPLE_RATE)
    #play_obj.wait_done()
    # Save the convolved sound
    # librosa.output.write_wav(
    #     '../sounds/convolved_sounds/convolved_sound.wav', data, RESAMPLE_RATE, norm=True)

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
    source_loc = [[0, 0], [9, 9]]

    # location of agent (or microphone in this case)
    agent_loc = np.array([9, 8])
    print(agent_loc[0], agent_loc[1])
    # init_env(paths, agent_loc, source_loc, room_config)



