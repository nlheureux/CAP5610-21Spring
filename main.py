import random
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchvision
import torchvision.datasets as datasets
from scipy.io import wavfile
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import noisereduce as nr
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.io.wavfile import write

#######tensorboard --logdir C:\Users\chaki\PycharmProjects\GAN\logs
#######http://localhost:6006/

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

#use mgan

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, img_dim),
            nn.LeakyReLU(0.01),
            #nn.tanh(),
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
lr = 3e-4
z_dim = 64
image_dim = 250000 * 1
batch_size = 2
num_epochs = 20

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

D = [] # Dataset
waveformDummy, sample_rate = torchaudio.load('original/m (0).wav')
fixed_sample_rate = sample_rate
waveformDummy = len(waveformDummy[1])
#### size will be 2433600
for row in range(300):

    waveform, sr = torchaudio.load('original/m ('+str(row)+').wav')
    #if len(waveform[1]) < waveformDummy:
    #    waveformDummy = len(waveform[1])
    #resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=fixed_sample_rate)
    #audio_mono = resample_transform(waveform)
    waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
    #print(waveform_mono.size())
    waveform_mono = torch.narrow(waveform_mono, 1, 0, 250000)
    #print(ps.size())
    D.append((waveform_mono, str(row)))

dataset = D
random.shuffle(dataset)

X_train, y_train = zip(*dataset)

loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

toplotG = []
toplotD = []
ep = list(range(1, num_epochs+1))

for epoch in range(num_epochs):
    for batch_idx, (real) in enumerate(loader):
        real = real.view(-1, 250000).to(device) #16384 is flattened array(128*128)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        #print('noise shape ', len(noise[0]))
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )
            toplotG.append((lossG))
            toplotD.append((lossD))

            with torch.no_grad():
                fake = gen(fixed_noise)
                # perform noise reduction
                #torchaudio.save(str(epoch)+'.wav', fake.cpu(), fixed_sample_rate)
                #data, rate = librosa.load(str(epoch)+'.wav')
                #noisy_part = data[0:250000]
                #reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=True)
                #write(str(epoch)+'rn.wav', rate, reduced_noise)
                #fake = fake[1]
                #print(fake.size())
                #output_signal = tf.audio.encode_wav(fake, fixed_sample_rate)

                #noisy_part = fake[1:250000]
                #reduced_noise = nr.reduce_noise(audio_clip=fake.numpy(), noise_clip=noisy_part.numpy(), verbose=True)
                #reduced_noise = tf.convert_to_tensor(reduced_noise, dtype=tf.float32)
                #song = np.array([reduced_noise],[reduced_noise])
                #torchaudio.save(str(epoch)+'.wav', fake, fixed_sample_rate)

                #data = real.reshape(-1, 1, 500, 500)

                #librosa.display.specshow(ps2, y_axis='mel', x_axis='time')
                #plt.show()

                step += 1

plt.plot(ep, toplotG)
plt.xlabel('Epoch')
plt.ylabel('Generator loss')
plt.show()

plt.plot(ep, toplotD)
plt.xlabel('Epoch')
plt.ylabel('Discriminator loss')
plt.show()

########output_signal = tf.audio.encode_wav(input_signal[0], input_signal[1])
