import os
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from src.dataset_mri import MRIDataset
from src.models_gated_attention2 import Generator

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips


def test():
    MODEL_PATH = "./checkpoints_ultra_finalNew6/ultra_final_epoch_59.pth"
    TEST_DATA_PATH = "/raid/s188598/comodGAN/brain_tumor_dataset/Testing"
    OUTPUT_DIR = "./resultsNew6/ultra_fft_50mask"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    netG = Generator().to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if 'ema_netG_state_dict' in checkpoint:
        netG.load_state_dict(checkpoint['ema_netG_state_dict'])
    elif 'netG_state_dict' in checkpoint:
        netG.load_state_dict(checkpoint['netG_state_dict'])
    else:
        netG.load_state_dict(checkpoint)

    netG.eval()

    dataset = MRIDataset(root_dir=TEST_DATA_PATH, img_size=256)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    total_ssim = 0
    total_lpips = 0
    count = 0

    with torch.no_grad():
        for i, (masked_img, mask, real_img, label) in enumerate(dataloader):
            masked_img = masked_img.to(device)
            mask = mask.to(device)
            real_img = real_img.to(device)

            fake_img = netG(masked_img, mask)
            comp_img = real_img * mask + fake_img * (1 - mask)

            real_norm = (real_img + 1) / 2
            comp_norm = (comp_img + 1) / 2

            total_ssim += ssim_metric(comp_norm, real_norm).item()
            total_lpips += lpips_fn(comp_img, real_img).item()

            real_uint8 = (real_norm * 255).clamp(0, 255).type(torch.uint8)
            comp_uint8 = (comp_norm * 255).clamp(0, 255).type(torch.uint8)
            fid_metric.update(real_uint8, real=True)
            fid_metric.update(comp_uint8, real=False)

            if i % 5 == 0:
                comparison = torch.cat([real_img, masked_img, comp_img], dim=3)
                vutils.save_image(comparison, f"{OUTPUT_DIR}/ultra_res_{i}.png", normalize=True)

            count += 1

    avg_ssim = total_ssim / count
    avg_lpips = total_lpips / count
    final_fid = fid_metric.compute().item()

    print("\n WYNIKI:")
    print(f"SSIM:  {avg_ssim:.4f}")
    print(f"LPIPS: {avg_lpips:.4f}")
    print(f"FID:  {final_fid:.4f}")

    with open(f"{OUTPUT_DIR}/report_ultra_v6.txt", "w") as f:
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"SSIM: {avg_ssim}\nLPIPS: {avg_lpips}\nFID: {final_fid}\n")



if __name__ == "__main__":
    test()