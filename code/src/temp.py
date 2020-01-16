# import torch
#
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name())
# print(torch.cuda.is_available())

import numpy as np
import matplotlib.pyplot as plt
from pyroomacoustics import ShoeBox, Beamformer, linear_2D_array

# Create a 4 by 6 metres shoe box room
room = ShoeBox([4, 6])

# Add a source somewhere in the room
room.add_source([2.5, 4.5])

# Create a linear array beamformer with 4 microphones
# with angle 0 degrees and inter mic distance 10 cm
R = linear_2D_array([2, 1.5], 4, 0, 0.1)
room.add_microphone_array(Beamformer(R, room.fs))

# Now compute the delay and sum weights for the beamformer
room.mic_array.rake_delay_and_sum_weights(room.sources[0][:1])

# plot the room and resulting beamformer
room.plot(freq=[1000, 2000, 4000, 8000], img_order=0)
plt.show()
