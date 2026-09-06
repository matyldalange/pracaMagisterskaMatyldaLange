# podejście piwerwsze "pierwotne" (małe maski)

import os
import torch
from torch.utils.data import DataLoader
from src.dataset_mri import MRIDataset
from src.models_mdgan import Generator, Discriminator
from src.losses_vgg19 import InpaintingLoss

DATASET_PATH = "/raid/s188598/comodGAN/brain_tumor_dataset/Training"
SAVE_PATH = "./checkpoints"
os.makedirs(SAVE_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50
batch_size = 8

dataset = MRIDataset(root_dir=DATASET_PATH, img_size=256)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

netG = Generator().to(device)
netD = Discriminator().to(device)
criterion_total = InpaintingLoss().to(device)
criterion_gan = torch.nn.MSELoss()

optG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(epochs):
    for i, (masked_img, mask, real_img) in enumerate(dataloader):
        masked_img, mask, real_img = masked_img.to(device), mask.to(device), real_img.to(device)

        optG.zero_grad()
        coarse_out, fine_out = netG(masked_img, mask)

        loss_recon = criterion_total(fine_out, real_img) + criterion_total(coarse_out, real_img)
        loss_g_adv = criterion_gan(netD(fine_out), torch.ones_like(netD(fine_out)))

        g_loss = loss_recon + 0.1 * loss_g_adv
        g_loss.backward()
        optG.step()

        optD.zero_grad()
        d_real_loss = criterion_gan(netD(real_img), torch.ones_like(netD(real_img)))
        d_fake_loss = criterion_gan(netD(fine_out.detach()), torch.zeros_like(netD(fine_out)))
        d_loss = (d_real_loss + d_fake_loss) * 0.5
        d_loss.backward()
        optD.step()

    torch.save({
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optG_state_dict': optG.state_dict(),
    }, f"{SAVE_PATH}/mdgan_epoch_{epoch}.pth")

    print(f"epoka {epoch} zakończona")