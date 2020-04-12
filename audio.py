# import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
# import librosa.display
import librosa.display

audio_path = 'Violin-Theme-MassTamilan.wav'
x , sr = librosa.load(audio_path)
print(type(x), type(sr))
print(x.shape, sr)
# ipd.Audio(audio_path)

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)