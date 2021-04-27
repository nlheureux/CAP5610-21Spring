import random
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import os
import numpy as np
import soundfile as sf

FILE_PATH = 'MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav'
FOLDER_PATH = '/volumes/External Hardrive/maestro-v3.0.0/'
ITERATIONS = 5


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
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        return self.gen(x)


# Hyper-parameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 128 * 128 * 1  # 16384
batch_size = 32
num_epochs = 100

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ]
)

dataset = []
directory = FOLDER_PATH + '2004/' + FILE_PATH
y, sr = librosa.load(directory)

a = librosa.to_mono(y)
dataset.append((np.abs(librosa.core.spectrum.stft(a))))

i = 0
# for subdir, dirs, files in os.walk(FOLDER_PATH):
#     for filename in files:
#         if i == ITERATIONS:
#             break
#         if filename.endswith(".wav"):
#             print('wav', filename)
#             y, sr = librosa.load(subdir + '/' + filename)
#             a = librosa.to_mono(y)
#             dataset.append((np.abs(librosa.core.spectrum.stft(a))))
#             print('audio', a)
#             i += 1
#         else:
#             continue
print('made it', dataset)
random.shuffle(dataset)

X_train, y_train = zip(*dataset)
X_train = np.array([x.reshape((128, 128, 1)) for x in X_train])

loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real) in enumerate(loader):
        real = real.view(-1, 16384).to(device)
        batch_size = real.shape[0]
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()
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

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 128, 128)
                audio_signal = librosa.core.spectrum.griffinlim(fake)
                sf.write('fake.wav', audio_signal, sr)
                data = real.reshape(-1, 1, 128, 128)

                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
