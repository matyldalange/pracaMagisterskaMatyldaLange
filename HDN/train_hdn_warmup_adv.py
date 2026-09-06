# podejście czwarte

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset_hdn import MRIDataset
from src.models_hdn_dilated_bottleneck import HDNGenerator, PatchGANDiscriminatorNew
from src.losses_vgg3_style_l1 import InpaintingLossNew

DATASET_PATH = "/raid/s188598/comodGAN/brain_tumor_dataset/Training"
SAVE_PATH = "./checkpoints_v3_Final"
os.makedirs(SAVE_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 16
LR = 2e-4
# ustawiamy liczbę epok rozgrzewkowych bez komponentu adversarial
WARM_UP_EPOCHS = 15

dataset = MRIDataset(root_dir=DATASET_PATH, img_size=256)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

netG = HDNGenerator().to(device)
netD = PatchGANDiscriminatorNew().to(device)

criterion_total = InpaintingLossNew().to(device)
criterion_gan = nn.BCEWithLogitsLoss().to(device)

optG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))
optD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))


for epoch in range(EPOCHS):
    netG.train()
    netD.train()

    # dynamiczne wyznaczanie wagi straty GAN
    if epoch < WARM_UP_EPOCHS:
        adv_w = 0.0
    else:
        adv_w = min(0.05, 0.005 * (epoch - WARM_UP_EPOCHS + 1))

    for i, (masked_img, mask, real_img, _) in enumerate(dataloader):
        masked_img = masked_img.to(device)
        mask = mask.to(device)
        real_img = real_img.to(device)
        inv_mask = 1 - mask

        optG.zero_grad()

        fine_out, _ = netG(masked_img, mask)

        loss_in = criterion_total(fine_out * inv_mask, real_img * inv_mask)
        loss_out = criterion_total(fine_out * mask, real_img * mask)
        loss_recon = 5.0 * loss_in + 1.0 * loss_out

        if adv_w > 0:
            g_fake_pred = netD(fine_out)
            loss_g_adv = criterion_gan(g_fake_pred, torch.ones_like(g_fake_pred))
        else:
            loss_g_adv = 0

        total_g_loss = loss_recon + adv_w * loss_g_adv
        total_g_loss.backward()
        optG.step()

        loss_d = torch.tensor(0.0).to(device)
        if adv_w > 0:
            optD.zero_grad()

            # zastosowanie Label Smoothing
            d_real = netD(real_img)
            loss_d_real = criterion_gan(d_real, torch.full_like(d_real, 0.9))

            d_fake = netD(fine_out.detach())
            loss_d_fake = criterion_gan(d_fake, torch.zeros_like(d_fake))

            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optD.step()

    print(
        f"Epoch [{epoch}/{EPOCHS}] | G_Loss: {total_g_loss.item():.4f} | D_Loss: {loss_d.item():.4f} | Adv_W: {adv_w:.4f}")

    if epoch % 5 == 0:
        save_file = os.path.join(SAVE_PATH, f"hdn_v3_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optG_state_dict': optG.state_dict(),
        }, save_file)
