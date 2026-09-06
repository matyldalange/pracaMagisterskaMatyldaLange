# podejście trzecie

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset_hdn import MRIDataset
from src.models_patchgan_instancenorm import HDNGenerator, PatchGANDiscriminatorNew
from src.losses_vgg_multi_mask_weighted import InpaintingLossNew

DATASET_PATH = "/raid/s188598/comodGAN/brain_tumor_dataset/Training"
SAVE_PATH = "./checkpointsFinalPoprawione"
os.makedirs(SAVE_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 16
LR = 2e-4

dataset = MRIDataset(root_dir=DATASET_PATH, img_size=256)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

netG = HDNGenerator().to(device)
netD = PatchGANDiscriminatorNew().to(device)

criterion_total = InpaintingLossNew().to(device)
criterion_gan = nn.BCEWithLogitsLoss().to(device)

optG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))
optD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))


for epoch in range(EPOCHS):
    for i, (masked_img, mask, real_img, _) in enumerate(dataloader):
        masked_img, mask, real_img = masked_img.to(device), mask.to(device), real_img.to(device)
        inv_mask = 1 - mask

        optD.zero_grad()
        d_real = netD(real_img)
        loss_d_real = criterion_gan(d_real, torch.ones_like(d_real))

        _, fine_out = netG(masked_img, mask)
        d_fake = netD(fine_out.detach())
        loss_d_fake = criterion_gan(d_fake, torch.zeros_like(d_fake))

        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        optD.step()

        optG.zero_grad()
        coarse_out, fine_out = netG(masked_img, mask)

        loss_in = criterion_total(fine_out * inv_mask, real_img * inv_mask)
        loss_out = criterion_total(fine_out * mask, real_img * mask)

        # podział straty na wnętrze ubytku (waga 5.0) oraz tło (waga 1.0)
        loss_recon_fine = 5.0 * loss_in + 1.0 * loss_out
        loss_recon_coarse = criterion_total(coarse_out, real_img)
        g_fake_pred = netD(fine_out)
        loss_g_adv = criterion_gan(g_fake_pred, torch.ones_like(g_fake_pred))

        total_g_loss = 0.5 * loss_recon_coarse + 1.0 * loss_recon_fine + 0.05 * loss_g_adv


        g_fake_pred = netD(fine_out)
        loss_g_adv = criterion_gan(g_fake_pred, torch.ones_like(g_fake_pred))

        total_g_loss.backward()
        optG.step()

    if epoch % 5 == 0:
        print(f"[Epoch {epoch}/{EPOCHS}] G_loss: {total_g_loss.item():.4f} D_loss: {loss_d.item():.4f}")
        torch.save({'netG_state_dict': netG.state_dict()}, f"{SAVE_PATH}/hdn_final_{epoch}.pth")