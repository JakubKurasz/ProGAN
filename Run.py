import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from math import log2
from PreProcessing import preProcessing
from Model import Generator
from Model import Discriminator
from Model import gradient_penalty

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 16, 16, 16, 8, 4, 4]
PROGRESSIVE_EPOCHS = [30,50,50,150,200,200,200,200]
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 500
FEATURES_CRITIC = 512
FEATURES_GEN = 512
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
START_TRAIN_AT_IMG_SIZE = 4

def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_step,
    scaler_gen,
    scaler_critic
):
    for batch_idx, (real, labels) in enumerate(train_dl):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = labels.to(device)

        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, 100, 1, 1).to(device)

        with torch.cuda.amp.autocast():
            fake = gen(noise,labels, alpha, step)
            critic_real = critic(real,labels, alpha, step)
            critic_fake = critic(fake.detach(),labels, alpha, step)
            gp = gradient_penalty(critic,labels, real, fake, alpha, step, device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake,labels, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
            (PROGRESSIVE_EPOCHS[step] * 0.5) * dataset
        )
        alpha = min(alpha, 1)
        if batch_idx % 44 == 0 and batch_idx > 0:
            with torch.no_grad():
                fake = gen(fixed_noise,labels,alpha, step) * 0.5 + 0.5
                # take out (up to) 32 examples
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)


                output_path = "fake_image.png"
                torchvision.utils.save_image(img_grid_fake, output_path)
            tensorboard_step += 1

    return tensorboard_step, alpha

Sizes2 = [[4,4],[8,6],[17,6],[34,12],[68,25],[137,50],[273,100],[546,199]]


# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(Z_DIM, FEATURES_GEN, CHANNELS_IMG).to(device)
critic = Discriminator(Z_DIM, FEATURES_CRITIC, CHANNELS_IMG).to(device)


# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

scaler_critic = torch.cuda.amp.GradScaler()
scaler_gen = torch.cuda.amp.GradScaler()

# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)


step = 0

gen.train()
critic.train()

tensorboard_step = 0
    # start at step that corresponds to img size that we set in config
step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))
for num_epochs in PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5  # start with very low alpha
        print(Sizes2[step])
        train_dl, test_dl, dataset = preProcessing(Sizes2[step])

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            tensorboard_step, alpha = train_fn(
                critic,
                gen,
                train_dl,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                scaler_gen,
                scaler_critic,
            )

        step += 1  # progress to the next img size

