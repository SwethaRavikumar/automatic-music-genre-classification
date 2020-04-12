import numpy as np
import wave as wv
import struct
import matplotlib.pyplot as plot

frequency = 1000
no_samples = 48000
pi_value = np.pi
sam_rate = 48000.0
amplitude = 16000
file = 'Violin-Theme-MassTamilan.wav'

sine_wave = [np.sin(2*pi_value*frequency*x/sam_rate) for x in range(no_samples)]
frames = no_samples
no_channels = 1
sampwidth = 2
comptype = "none"
compname = "not compressed"

wavefile = wv.open(file,'w')
wavefile.setparams((no_channels,sampwidth, int(sam_rate),comptype,compname))
for s in sine_wave:
    wavefile.writeframes(struct.pack('h',int(s*amplitude)))