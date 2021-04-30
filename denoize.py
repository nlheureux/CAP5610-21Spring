import librosa
from scipy.io import wavfile
import noisereduce as nr
from scipy.io.wavfile import write

data, rate = librosa.load('1l.wav')
print(data.shape)
print(rate)
#data = data[0:1,0:264192]
#print(data.shape)
noisy_part = data[0:250000]
print(noisy_part.shape)
reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=True)
write('1lrn.wav', rate, reduced_noise)