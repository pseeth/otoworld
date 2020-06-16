import numpy as np
import torch


def ipd_ild_features(stft_data, sampling_rate=8000):
    """
    Computes interphase difference (IPD) and interlevel difference (ILD) for a
    stereo spectrogram.
    Args:
       stft_data (Torch tensor): Tensor of shape batch_size, time_frames, mag+phase, channels, sources
       sampling_rate (int): The rate at which data is sampled. Default = 8000 Hz
    Returns:
       ipd (``Torch tensor`): Interphase difference between two channels
       ild (``Torch tensor``): Interlevel difference between two channels
    """

    # The mag and phase are concatenated together, so each of them will have half the dimensionality
    mag_phase_dim = stft_data.shape[2]//2

    # Separate the data by channels
    stft_ch_one = stft_data[:, :, :, 0]
    stft_ch_two = stft_data[:, :, :, 1]

    # Calculate ILD over the magnitudes

    # Extract the magnitudes from the stft data
    stft_ch_one_mag = stft_ch_one[:, :, 0:mag_phase_dim]
    stft_ch_two_mag = stft_ch_two[:, :, 0:mag_phase_dim]
    vol = torch.abs(stft_ch_one_mag) + torch.abs(stft_ch_two_mag)
    ild = torch.abs(stft_ch_one_mag) / (torch.abs(stft_ch_two_mag) + 1e-4)
    ild = 20 * torch.log10(ild + 1e-8)

    # Extract the phase from the stft data
    phase_ch_two = stft_ch_one[..., -mag_phase_dim:]
    phase_ch_one = stft_ch_two[..., -mag_phase_dim:]

    ipd = torch.fmod(phase_ch_two - phase_ch_one, np.pi)

    # Output shape of ILD and IPD = [batch_size, time_frames, mag_phase_dim, sources]
    return ipd, ild, vol

