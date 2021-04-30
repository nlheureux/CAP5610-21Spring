import random


import librosa.display
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import soundfile as sf

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


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, img_dim),
            nn.LeakyReLU(0.01),
            nn.Tanh(),
            #nn.Sigmoid(),
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = (221400) #128 * 128 * 1  # 16384
batch_size = 1
num_epochs = 20

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

#print(sr)
#librosa.display.specshow(ps, y_axis='mel', x_axis='time')
#plt.show()

D = [] # Dataset

for row in range(1000, 1250):
    y, sr = librosa.load('original/m ('+str(row)+').wav', duration=5)
    #print (sr)
    #print(y.shape)
    a = librosa.to_mono(y)
    abs_spectrogram = np.abs(librosa.core.spectrum.stft(a))
    #print(abs_spectrogram.shape)
    #ps = librosa.feature.melspectrogram(y=y, sr=sr)
    #print(ps.shape)
    #if ps.shape != (128, 128): continue
    D.append((abs_spectrogram))



dataset = D
random.shuffle(dataset)

#X_train = zip(*dataset)
X_train = dataset
#print(X_train.shape)
#X_train = np.array([x.reshape( (1025, 388, 1) ) for x in X_train])

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
        real = real.view(-1, 221400).to(device) #16384
        #print(real.shape)
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
                print(fake.size())
                #fake = fake[0].reshape(-1, 1, 1025, 216)
                fake = fake[0]
                print(fake.size())
                fake = fake.cpu().numpy()
                print(fake)
                fake = fake.reshape((1025, 216))
                audio_signal = librosa.core.spectrum.griffinlim(fake)
                #print(fake.size())
                sf.write(str(epoch)+'l.wav', audio_signal, 22050)
                #data = real.reshape(-1, 1, 128, 128)

                #librosa.output.write_wav('/generated/test', fake, sr, norm=False)
                #sf.write('new_file.flac', fake.cpu(), 22050)
                #ps2=data[step][0]
                #print(data.size())
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
    #librosa.display.specshow(ps2, y_axis='mel', x_axis='time')
    #plt.show()
    #librosa.output.write_wav('/generated/test', ps2[0], ps2[1], norm=False)

########output_signal = tf.audio.encode_wav(input_signal[0], input_signal[1])