from scipy.io import wavfile
import torchaudio
import csv
import torch

s=0
e=5
x=0
with open('dataset.csv', 'w', newline='') as file:
    for r in range (25):
        sampleRate, waveform = wavfile.read("2.wav")
        start = int(s * sampleRate)
        end = int(e * sampleRate)
        wavfile.write("sampled/"+str(x)+".wav", sampleRate, waveform[start:end])
        waveform, sample_rate = torchaudio.load('sampled/'+str(r)+'.wav')
        fixed_sample_rate = 22050
        resample_transform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=fixed_sample_rate)
        audio_mono = resample_transform(waveform)
        torch.set_printoptions(profile="full")
        writer = csv.writer(file, delimiter=";")
        if (r == 0):
            writer.writerow(["waveform"])
        writer.writerow([waveform[0]])
        s=s+5
        e=e+5
        x=x+1
