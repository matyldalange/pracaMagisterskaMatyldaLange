# podejście ósme (wieksze maski - siódme)

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset_mri import MRIDataset
from src.models_gated_attention2 import Generator, MultiScaleDiscriminator
from src.losses_mri_hinge import InpaintingLoss


class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)


def train():
    DATASET_PATH = "/raid/s188598/comodGAN/brain_tumor_dataset/Training"
    SAVE_PATH = "./checkpoints_ultra_finalNew7"
    os.makedirs(SAVE_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    batch_size = 16

    # osobne tempo uczenia dla generatora i dyskryminatora
    lr_G = 1e-4
    lr_D = 4e-4

    dataset = MRIDataset(root_dir=DATASET_PATH, img_size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    netG = Generator().to(device)
    netD = MultiScaleDiscriminator().to(device)
    ema_g = EMAModel(netG, decay=0.9999)

    criterion_total = InpaintingLoss().to(device)
    l1_loss = nn.L1Loss()

    optG = torch.optim.Adam(netG.parameters(), lr=lr_G, betas=(0.0, 0.9))
    optD = torch.optim.Adam(netD.parameters(), lr=lr_D, betas=(0.0, 0.9))


    for epoch in range(epochs):
        netG.train()
        netD.train()

        for i, (masked_img, mask, real_img, label) in enumerate(dataloader):
            masked_img, mask, real_img = masked_img.to(device), mask.to(device), real_img.to(device)

            optG.zero_grad()
            fake_img = netG(masked_img, mask)
            comp_img = real_img * mask + fake_img * (1 - mask)

            inv_mask = 1.0 - mask
            loss_global = criterion_total(comp_img, real_img)
            loss_local = criterion_total(comp_img * inv_mask, real_img * inv_mask)
            loss_recon = loss_global + 2.0 * loss_local

            d1_fake, d2_fake = netD(comp_img)
            d1_real, d2_real = netD(real_img)

            # Hinge Loss dla Generatora
            loss_g_adv = -torch.mean(d1_fake) - torch.mean(d2_fake)

            loss_fm = l1_loss(d1_fake, d1_real.detach()) + l1_loss(d2_fake, d2_real.detach())

            g_loss = loss_recon + 0.1 * loss_g_adv + 1.0 * loss_fm

            g_loss.backward()
            optG.step()
            ema_g.update(netG)

            optD.zero_grad()

            d1_real_d, d2_real_d = netD(real_img)
            d1_fake_d, d2_fake_d = netD(comp_img.detach())

            # Hinge Loss dla Dyskryminatora
            loss_d_real = torch.mean(F.relu(1.0 - d1_real_d)) + torch.mean(F.relu(1.0 - d2_real_d))

            loss_d_fake = torch.mean(F.relu(1.0 + d1_fake_d)) + torch.mean(F.relu(1.0 + d2_fake_d))

            loss_d = (loss_d_real + loss_d_fake) * 0.5

            loss_d.backward()
            optD.step()

        checkpoint_name = f"{SAVE_PATH}/ultra_final_epoch_{epoch}.pth"
        torch.save({
            'netG_state_dict': netG.state_dict(),
            'ema_netG_state_dict': ema_g.ema_model.state_dict(),
            'netD_state_dict': netD.state_dict(),
        }, checkpoint_name)

        print(f"epoka {epoch}/{epochs} zakończona")


if __name__ == "__main__":
    train()