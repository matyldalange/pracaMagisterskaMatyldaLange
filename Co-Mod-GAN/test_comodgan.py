import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from argparse import Namespace
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

from models.networks.co_mod_gan import Generator
from mri_dataset import MRICoModDataset


def run_testing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    DATA_PATH = "/raid/s188598/comodGAN/brain_tumor_dataset/Testing"
    CHECKPOINT = "checkpoint_new_epoch_95.pth"

    # inicjalizacja metryk LPIPS, SSIM, FID
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    dataset = MRICoModDataset(DATA_PATH, size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    netG = Generator(opt=opt).to(device)
    if os.path.exists(CHECKPOINT):
        netG.load_state_dict(torch.load(CHECKPOINT, map_location=device))
        netG.eval()
    else:
        return

    ssim_values = []
    lpips_values = []

    os.makedirs("resultsNew", exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Testowanie")):
            real_img = batch['gt'].to(device)
            masked_img = batch['masked'].to(device)
            mask = batch['mask'].to(device)

            if real_img.shape[1] == 1:
                real_img = real_img.repeat(1, 3, 1, 1)
            if masked_img.shape[1] == 1:
                masked_img = masked_img.repeat(1, 3, 1, 1)

            z = torch.randn(real_img.size(0), LATENT_DIM).to(device)
            fake_img, _, _ = netG(masked_img, mask, [z])

            # denormalizacja z [-1, 1] do [0, 1] do obliczenia SSIM oraz LPIPS
            real_norm = (real_img + 1) / 2.0
            fake_norm = (fake_img + 1) / 2.0

            real_norm = torch.clamp(real_norm, 0, 1)
            fake_norm = torch.clamp(fake_norm, 0, 1)

            ssim_values.append(ssim_metric(fake_norm, real_norm).item())
            lpips_values.append(lpips_metric(fake_norm, real_norm).item())

            real_uint8 = (real_norm * 255).byte()
            fake_uint8 = (fake_norm * 255).byte()
            fid_metric.update(real_uint8, real=True)
            fid_metric.update(fake_uint8, real=False)

            # zapis ptzykładowych efektów wizualnych
            if i % 10 == 0:
                vutils.save_image(torch.cat([masked_img, fake_img, real_img], dim=3),
                                  f"resultsNew/sample_{i}.png", normalize=True)

    final_ssim = np.mean(ssim_values)
    final_lpips = np.mean(lpips_values)
    final_fid = fid_metric.compute().item()

    print(f"\n WYNIKI:")
    print(f"SSIM:  {final_ssim:.4f} ")
    print(f"LPIPS: {final_lpips:.4f} ")
    print(f"FID:   {final_fid:.4f} ")


if __name__ == "__main__":
    run_testing()