# train_model.py
# Script to train a Conditional GAN on MNIST.
# This version includes Dropout in the Generator to prevent mode collapse
# and improve the variety of generated images.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Generator network (with Dropout to prevent mode collapse)
class Generator(nn.Module):
    def __init__(self, latent_dim, class_dim):
        super().__init__()
        self.label_emb = nn.Embedding(10, class_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + class_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.4),  # <-- ADDED DROPOUT
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.4),  # <-- ADDED DROPOUT
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate noise vector and label embedding
        c = self.label_emb(labels)
        x = torch.cat([noise, c], dim=1)
        # Generate image
        return self.model(x).view(-1, 1, 28, 28)

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, class_dim):
        super().__init__()
        self.label_emb = nn.Embedding(10, class_dim)
        self.model = nn.Sequential(
            nn.Linear(784 + class_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        # Concatenate image and label embedding
        c = self.label_emb(labels)
        x = torch.cat([img_flat, c], dim=1)
        # Classify
        return self.model(x)

if __name__ == '__main__':
    # Hyperparameters
    latent_dim = 100
    class_dim = 10
    batch_size = 128
    epochs = 50  # 50 epochs is a good starting point for decent results
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1] for Tanh activation
    ])
    dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize models
    gen = Generator(latent_dim, class_dim).to(device)
    disc = Discriminator(class_dim).to(device)

    # Loss function and optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, beta2))

    # --- Training Loop ---
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(loader):
            # Adversarial ground truths
            valid = torch.full((batch_size, 1), 1.0, device=device, requires_grad=False)
            fake = torch.full((batch_size, 1), 0.0, device=device, requires_grad=False)

            # Configure input
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = torch.randn(batch_size, latent_dim, device=device)

            # Generate a batch of images
            gen_imgs = gen(z, labels)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(disc(gen_imgs, labels), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss for real images
            real_loss = adversarial_loss(disc(real_imgs, labels), valid)

            # Loss for fake images (detach to avoid backprop through generator)
            fake_loss = adversarial_loss(disc(gen_imgs.detach(), labels), fake)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()
            
            if (i+1) % 200 == 0:
                print(
                    f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(loader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )

    # Save the trained generator
    print("Training finished. Saving model...")
    torch.save(gen.state_dict(), 'generator.pth')
    print("Model 'generator.pth' saved successfully.")