import os
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from src.dataset_mri import MRIDataset
from src.models_biggest_mask import Generator

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips


def test():
    MODEL_PATH = "./checkpoints_controlled_erasure/erasure_epoch_59.pth"
    TEST_DATA_PATH = "/raid/s188598/comodGAN/brain_tumor_dataset/Testing"
    OUTPUT_DIR = "./results_erasure"
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

    total_ssim = 0
    total_lpips = 0
    count = 0

    with torch.no_grad():
        for i, (masked_img, mask, real_img, label) in enumerate(dataloader):
            real_img = real_img.to(device)
            mask = mask.to(device)

            fake_img = netG(real_img, mask)

            erased_result = real_img * (1.0 - mask) + fake_img * mask

            real_norm = (real_img + 1) / 2
            comp_norm = (erased_result + 1) / 2

            total_ssim += ssim_metric(comp_norm, real_norm).item()
            total_lpips += lpips_fn(erased_result, real_img).item()

            real_uint8 = (real_norm * 255).clamp(0, 255).type(torch.uint8)
            comp_uint8 = (comp_norm * 255).clamp(0, 255).type(torch.uint8)
            fid_metric.update(real_uint8, real=True)
            fid_metric.update(comp_uint8, real=False)

            if i % 5 == 0:
                visual_mask = mask.repeat(1, 3, 1, 1) * 2 - 1
                comparison = torch.cat([real_img, real_img * (1.0 - mask), erased_result], dim=3)
                vutils.save_image(comparison, f"{OUTPUT_DIR}/erasure_sample_{i}.png", normalize=True)

            count += 1

    avg_ssim = total_ssim / count
    avg_lpips = total_lpips / count
    final_fid = fid_metric.compute().item()

    print("WYNIKI:")
    print(f"SSIM:  {avg_ssim:.4f}")
    print(f"LPIPS: {avg_lpips:.4f}")
    print(f"FID:  {final_fid:.4f}")


if __name__ == "__main__":
    test()