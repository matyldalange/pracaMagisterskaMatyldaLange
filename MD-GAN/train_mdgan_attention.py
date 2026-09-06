# podejście drugie (wieksze maski - podejście pierwsze dla wiekszych masek)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset_mri import MRIDataset
from src.models_mdgan_attention import Generator, Discriminator
from src.losses_attention_edge import InpaintingLoss


def train():
    DATASET_PATH = "/raid/s188598/comodGAN/brain_tumor_dataset/Training"
    SAVE_PATH = "./checkpoints_newMask"
    os.makedirs(SAVE_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50
    batch_size = 8
    lr = 2e-4
    beta1, beta2 = 0.5, 0.999

    dataset = MRIDataset(root_dir=DATASET_PATH, img_size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    netG = Generator().to(device)
    netD = Discriminator().to(device)

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

            coarse_out, fine_out = netG(masked_img, mask)

            loss_recon = criterion_total(fine_out, real_img) + 0.5 * criterion_total(coarse_out, real_img)

            pred_fake = netD(fine_out)
            loss_g_adv = criterion_gan(pred_fake, torch.ones_like(pred_fake))

            g_loss = loss_recon + 0.1 * loss_g_adv

            g_loss.backward()
            optG.step()

            optD.zero_grad()

            pred_real = netD(real_img)
            d_real_loss = criterion_gan(pred_real, torch.ones_like(pred_real))

            pred_fake_d = netD(fine_out.detach())
            d_fake_loss = criterion_gan(pred_fake_d, torch.zeros_like(pred_fake_d))

            d_loss = (d_real_loss + d_fake_loss) * 0.5

            d_loss.backward()
            optD.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[G loss: {g_loss.item():.4f}] [D loss: {d_loss.item():.4f}]")

        checkpoint_name = f"{SAVE_PATH}/mdgan_new_epoch__mask{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optG_state_dict': optG.state_dict(),
            'optD_state_dict': optD.state_dict(),
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item()
        }, checkpoint_name)

        print(f"epoka {epoch} zakończona")


if __name__ == "__main__":
    train()