# podejście trzecie (większe maski - podejście drugie)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset_mri import MRIDataset
from src.models_gated_multiscale import Generator, MultiScaleDiscriminator
from src.losses_attention_edge import InpaintingLoss


def train():
    DATASET_PATH = "/raid/s188598/comodGAN/brain_tumor_dataset/Training"
    SAVE_PATH = "./checkpoints_New2"
    os.makedirs(SAVE_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 60
    batch_size = 16
    lr = 2e-4
    beta1, beta2 = 0.5, 0.999

    dataset = MRIDataset(root_dir=DATASET_PATH, img_size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    netG = Generator().to(device)
    netD = MultiScaleDiscriminator().to(device)

    criterion_total = InpaintingLoss().to(device)
    criterion_gan = nn.MSELoss()

    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))


    for epoch in range(epochs):
        netG.train()
        netD.train()

        for i, (masked_img, mask, real_img, label) in enumerate(dataloader):
            masked_img = masked_img.to(device)
            mask = mask.to(device)
            real_img = real_img.to(device)

            optG.zero_grad()

            fake_img = netG(masked_img, mask)

            comp_img = real_img * mask + fake_img * (1 - mask)

            loss_recon = criterion_total(comp_img, real_img)

            d1_fake, d2_fake = netD(comp_img)

            loss_g_adv = (criterion_gan(d1_fake, torch.ones_like(d1_fake)) +
                          criterion_gan(d2_fake, torch.ones_like(d2_fake))) * 0.5

            g_loss = loss_recon + 0.1 * loss_g_adv

            g_loss.backward()
            optG.step()

            optD.zero_grad()

            d1_real, d2_real = netD(real_img)
            d1_fake_d, d2_fake_d = netD(comp_img.detach())

            loss_d_real = (criterion_gan(d1_real, torch.ones_like(d1_real)) +
                           criterion_gan(d2_real, torch.ones_like(d2_real))) * 0.5

            loss_d_fake = (criterion_gan(d1_fake_d, torch.zeros_like(d1_fake_d)) +
                           criterion_gan(d2_fake_d, torch.zeros_like(d2_fake_d))) * 0.5

            d_loss = (loss_d_real + loss_d_fake) * 0.5

            d_loss.backward()
            optD.step()

            if i % 20 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[G: {g_loss.item():.4f}] [D: {d_loss.item():.4f}] [Recon: {loss_recon.item():.4f}]")

        checkpoint_name = f"{SAVE_PATH}/mdgan_ultra_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optG_state_dict': optG.state_dict(),
            'optD_state_dict': optD.state_dict(),
        }, checkpoint_name)

        print(f"epoka {epoch} zakończona")


if __name__ == "__main__":
    train()