import random
import torch
import torch.nn as nn
import torchaudio
import torchvision
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import os
from music21 import converter, instrument, note, chord
from magenta.models.gansynth.lib import networks as network
from note_seq import midi_io as ns
from note_seq import midi_synth

FILE_PATH = 'MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav'
FOLDER_PATH = '/volumes/External Hardrive/maestro-v3.0.0/'
ITERATIONS = 1

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(500, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 2000),
            nn.LeakyReLU(0.01),
            nn.Linear(2000, img_dim),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters etc.
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
lr = 3e-4
z_dim = 64
image_dim = 250000 * 1
batch_size = 2
num_epochs = 20

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ]
)
test = ns.midi_file_to_note_sequence('Queen - Bohemian Rhapsody.mid')
#x = midi_synth.fluidsynth(test, 16000, None)
print(type(test))
directory = 'Queen - Bohemian Rhapsody.mid'
D = []  # Dataset
#waveformDummy, sample_rate = torchaudio.load(directory)
#fixed_sample_rate = sample_rate
#waveformDummy = len(waveformDummy[1])
#midi = converter.parse(directory)

i = 0
D = []  # Dataset
for subdir, dirs, files in os.walk(FOLDER_PATH):
    for filename in files:
        print(filename)
        if i == ITERATIONS:
            break
        if filename.endswith(".mid"):
            #midi = converter.parse(filename)
            #print(midi)
            i += 1
        else:
            continue
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
print('writer_real', writer_real)

for epoch in range(num_epochs):
    for batch_idx, (real) in enumerate(loader):
        real = real.view(-1, 250000).to(device)  # 16384 is flattened array(128*128)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
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

            with torch.no_grad():
                fake = gen(fixed_noise)
                torchaudio.save(str(epoch) + '.wav', fake, fixed_sample_rate)

                data = real.reshape(-1, 1, 500, 500)

                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
