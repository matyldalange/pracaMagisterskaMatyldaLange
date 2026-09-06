import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim


from models.networks.co_mod_gan import Generator
from models.networks.co_mod_gan import Discriminator
from mri_dataset import MRICoModDataset
from argparse import Namespace

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # konfiguracja parametrów architektury i optymalizacji
    opt = Namespace()
    opt.crop_size = 256
    opt.dlatent_size = 512
    opt.num_channels = 3
    opt.fmap_base = 16384
    opt.fmap_decay = 1.0
    opt.fmap_min = 1
    opt.fmap_max = 512
    opt.randomize_noise = True
    opt.architecture = 'skip'
    opt.nonlinearity = 'lrelu'
    opt.resample_kernel = [1, 3, 3, 1]
    opt.fused_modconv = True
    opt.pix2pix = False
    opt.dropout_rate = 0.5
    opt.cond_mod = True
    opt.style_mod = True
    opt.noise_injection = True
    opt.mbstd_num_features = 1
    opt.mbstd_group_size = 4

    IMG_SIZE = 256
    LATENT_DIM = 512
    BATCH_SIZE = 4
    LR = 0.0001
    EPOCHS = 100
    DATA_PATH = "/raid/s188598/comodGAN/brain_tumor_dataset/Training"

    dataset = MRICoModDataset(DATA_PATH, size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    netG = Generator(opt=opt).to(device)
    netD = Discriminator(opt=opt).to(device)

    opt_g = optim.Adam(netG.parameters(), lr=LR, betas=(0.0, 0.99))
    opt_d = optim.Adam(netD.parameters(), lr=LR, betas=(0.0, 0.99))

    for epoch in range(EPOCHS):
        for i, batch in enumerate(dataloader):
            real_img = batch['gt'].to(device)
            masked_img = batch['masked'].to(device)
            mask = batch['mask'].to(device)

            # dostosowanie kanałów do kompatybilności z  modelem
            if real_img.shape[1] == 1:
                real_img = real_img.repeat(1, 3, 1, 1)
            if masked_img.shape[1] == 1:
                masked_img = masked_img.repeat(1, 3, 1, 1)

            netD.zero_grad()

            z = torch.randn(real_img.size(0), LATENT_DIM).to(device)
            fake_img, _, _ = netG(masked_img, mask, [z])

            pred_real = netD(real_img, mask)
            pred_fake = netD(fake_img.detach(), mask)

            # strata adversarialna dla dyskryminatora
            loss_d = F.softplus(-pred_real).mean() + F.softplus(pred_fake).mean()
            loss_d.backward()
            opt_d.step()

            netG.zero_grad()
            pred_fake_g = netD(fake_img, mask)

            loss_g_adv = F.softplus(-pred_fake_g).mean()

            loss_l1 = F.l1_loss(fake_img, real_img) * 10.0

            total_g_loss = loss_g_adv + loss_l1

            total_g_loss.backward()
            opt_g.step()

            if i % 10 == 0:
                print(f"[{epoch}/{EPOCHS}][{i}/{len(dataloader)}] "
                      f"Loss_D: {loss_d.item():.4f} Loss_G: {total_g_loss.item():.4f} "
                      f"L1: {loss_l1.item():.4f}")

        # zapisywanie punktu kontrolnego co 5 epok
        if epoch % 5 == 0:
            torch.save(netG.state_dict(), f"checkpoint_new_epoch_{epoch}.pth")


if __name__ == "__main__":
    class Config:
        def __init__(self):
            self.crop_size = 512
            self.dlatent_size = 512
            self.num_channels = 3
            self.fmap_base = 16384
            self.fmap_decay = 1.0
            self.fmap_min = 1
            self.fmap_max = 512
            self.randomize_noise = True
            self.architecture = 'skip'
            self.nonlinearity = 'lrelu'
            self.resample_kernel = [1, 3, 3, 1]
            self.fused_modconv = True
            self.pix2pix = False
            self.dropout_rate = 0.5
            self.cond_mod = True
            self.style_mod = True
            self.noise_injection = True
            self.mbstd_num_features = 1
            self.mbstd_group_size = 4


    opt = Config()
    run_training()