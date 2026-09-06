import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

from src.dataset_hdn import MRIDataset
from src.models_patchgan_instancenorm import HDNGenerator

DATASET_PATH = "/raid/s188598/comodGAN/brain_tumor_dataset/Testing"
CHECKPOINT_PATH = "./checkpointsFinalPoprawione/hdn_final_95.pth"
RESULT_SAVE_PATH = "/raid/s188598/HDNulepszenia/test_resultsFinalPoprawione"
os.makedirs(RESULT_SAVE_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG = HDNGenerator().to(device)

if not os.path.exists(CHECKPOINT_PATH):
    import glob

    available = glob.glob("./checkpointsFinalPoprawione/*.pth")
    if available:
        CHECKPOINT_PATH = max(available, key=os.path.getctime)
    else:
        exit()

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
netG.load_state_dict(checkpoint['netG_state_dict'])
netG.eval()

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
fid_metric = FrechetInceptionDistance(feature=2048).to(device)

dataset = MRIDataset(root_dir=DATASET_PATH, img_size=256)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)


avg_ssim = 0
avg_lpips = 0
count = 0

with torch.no_grad():
    for i, (masked_img, mask, real_img, _) in enumerate(dataloader):
        masked_img, mask, real_img = masked_img.to(device), mask.to(device), real_img.to(device)

        _, fake_img = netG(masked_img, mask)

        fake_img_norm = (fake_img * 0.5 + 0.5).clamp(0, 1)
        real_img_norm = (real_img * 0.5 + 0.5).clamp(0, 1)
        masked_img_norm = (masked_img * 0.5 + 0.5).clamp(0, 1)

        avg_ssim += ssim_metric(fake_img_norm, real_img_norm)
        avg_lpips += lpips_metric(fake_img_norm, real_img_norm)

        real_uint8 = (real_img_norm * 255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        fake_uint8 = (fake_img_norm * 255).add_(0.5).clamp_(0, 255).to(torch.uint8)

        fid_metric.update(real_uint8, real=True)
        fid_metric.update(fake_uint8, real=False)

        if i < 10:
            comparison = torch.cat([real_img_norm, masked_img_norm, fake_img_norm], dim=3)
            vutils.save_image(comparison, f"{RESULT_SAVE_PATH}/final_eval_{i}.png")

        count += 1
        if i % 10 == 0:
            print(f"Przetworzono batch {i}")

if count > 0:
    res_ssim = (avg_ssim / count).item()
    res_lpips = (avg_lpips / count).item()
    res_fid = fid_metric.compute().item()

    print(f"\n WYNIKI:")
    print(f"SSIM:  {res_ssim:.5f}")
    print(f"LPIPS: {res_lpips:.5f}")
    print(f"FID:   {res_fid:.5f}")

    with open(f"{RESULT_SAVE_PATH}/final_report.txt", "w") as f:
        f.write(f"SSIM: {res_ssim}\nLPIPS: {res_lpips}\nFID: {res_fid}")
