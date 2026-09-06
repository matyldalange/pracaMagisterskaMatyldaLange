import os
import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from src.dataset_mri import MRIDataset
from src.models_biggest_mask import Generator

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips


def test():
    MODEL_PATH = "./checkpoints_controlled_erasure_v2/erasure_epoch_59.pth"
    TEST_DATA_PATH = "/raid/s188598/comodGAN/brain_tumor_dataset/Testing"
    OUTPUT_DIR = "./results_erasure_v2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    netG = Generator().to(device)


    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if 'ema_netG_state_dict' in checkpoint:
        netG.load_state_dict(checkpoint['ema_netG_state_dict'])
    else:
        netG.load_state_dict(checkpoint['netG_state_dict'])

    netG.eval()

    dataset = MRIDataset(root_dir=TEST_DATA_PATH, img_size=256)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_ssim_bg = 0
    total_ssim_full = 0
    total_lpips = 0
    count = 0

    with torch.no_grad():
        for i, (masked_img, mask, real_img, label) in enumerate(dataloader):
            real_img = real_img.to(device)
            mask = mask.to(device)

            smooth_mask = TF.gaussian_blur(mask, kernel_size=(9, 9), sigma=(2.0, 2.0))
            inv_mask = 1.0 - smooth_mask

            fake_img = netG(real_img, mask)
            erased_result = real_img * inv_mask + fake_img * smooth_mask

            real_norm = (real_img + 1) / 2
            comp_norm = (erased_result + 1) / 2

            total_ssim_bg += ssim_metric(comp_norm * inv_mask, real_norm * inv_mask).item()
            total_ssim_full += ssim_metric(comp_norm, real_norm).item()
            total_lpips += lpips_fn(erased_result, real_img).item()

            real_uint8 = (real_norm * 255).clamp(0, 255).type(torch.uint8)
            comp_uint8 = (comp_norm * 255).clamp(0, 255).type(torch.uint8)
            fid_metric.update(real_uint8, real=True)
            fid_metric.update(comp_uint8, real=False)

            if i % 5 == 0:
                comparison = torch.cat([real_img, real_img * inv_mask, erased_result], dim=3)
                vutils.save_image(comparison, f"{OUTPUT_DIR}/erasure_sample_{i}.png", normalize=True)

            count += 1

    avg_ssim_bg = total_ssim_bg / count
    avg_ssim_full = total_ssim_full / count
    avg_lpips = total_lpips / count
    final_fid = fid_metric.compute().item()

    print("\n WYNIKI:")
    print(f"SSIM:          {avg_ssim_full:.4f}")
    print(f"LPIPS:           {avg_lpips:.4f}")
    print(f"FID:            {final_fid:.4f}")

    with open(f"{OUTPUT_DIR}/report_erasure_v2.txt", "w") as f:
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"SSIM: {avg_ssim_full}\nLPIPS: {avg_lpips}\nFID: {final_fid}\n")


if __name__ == "__main__":
    test()