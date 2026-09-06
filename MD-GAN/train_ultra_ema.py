# podejście czwarte ( wieksze maski - podejście trzecie)

import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset_mri import MRIDataset
from src.models_ultra_ema import Generator, MultiScaleDiscriminator
from src.losses_attention_edge import InpaintingLoss


class EMAModel:
    def __init__(self, model, decay=0.999):
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
    SAVE_PATH = "./checkpoints_ultra_finalNew3"
    os.makedirs(SAVE_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 60
    batch_size = 16
    lr = 2e-4

    dataset = MRIDataset(root_dir=DATASET_PATH, img_size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    netG = Generator().to(device)
    netD = MultiScaleDiscriminator().to(device)
    ema_g = EMAModel(netG)

    criterion_total = InpaintingLoss().to(device)
    criterion_gan = nn.MSELoss()

    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (masked_img, mask, real_img, label) in enumerate(dataloader):
            masked_img, mask, real_img = masked_img.to(device), mask.to(device), real_img.to(device)

            optG.zero_grad()
            fake_img = netG(masked_img, mask)
            comp_img = real_img * mask + fake_img * (1 - mask)

            loss_recon = criterion_total(comp_img, real_img)

            d1_fake, d2_fake = netD(comp_img)
            loss_g_adv = (criterion_gan(d1_fake, torch.ones_like(d1_fake)) +
                          criterion_gan(d2_fake, torch.ones_like(d2_fake))) * 0.5

            g_loss = loss_recon + 0.15 * loss_g_adv
            g_loss.backward()
            optG.step()

            # aktualizacja EMA Generatora
            ema_g.update(netG)

            optD.zero_grad()
            d1_real, d2_real = netD(real_img)
            d1_fake_d, d2_fake_d = netD(comp_img.detach())

            loss_d = (criterion_gan(d1_real, torch.ones_like(d1_real)) +
                      criterion_gan(d1_fake_d, torch.zeros_like(d1_fake_d)) +
                      criterion_gan(d2_real, torch.ones_like(d2_real)) +
                      criterion_gan(d2_fake_d, torch.zeros_like(d2_fake_d))) * 0.25

            loss_d.backward()
            optD.step()

        checkpoint_name = f"{SAVE_PATH}/ultra_final_epoch_{epoch}.pth"
        torch.save({
            'netG_state_dict': netG.state_dict(),
            'ema_netG_state_dict': ema_g.ema_model.state_dict(),
            'netD_state_dict': netD.state_dict(),
        }, checkpoint_name)
        print(f"epoka {epoch} zakończona")


if __name__ == "__main__":
    train()