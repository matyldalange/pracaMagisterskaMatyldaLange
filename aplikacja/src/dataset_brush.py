import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from src.mask_generator import FreeFormMaskGenerator


class MRIBrushDataset(Dataset):
    def __init__(self, root_dir, img_size=256, max_coverage=0.5):
        self.root_dir = root_dir
        self.img_size = img_size
        self.max_coverage = max_coverage

        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*", "*.jpg"))) + \
                           sorted(glob.glob(os.path.join(root_dir, "*", "*.png")))

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.mask_gen = FreeFormMaskGenerator(img_size=img_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        real_img = self.transform(image)

        # generowanie losowej maski pędzla (do 50% powierzchni)
        mask = self.mask_gen(max_coverage=self.max_coverage)

        # obraz z wymazanym obszarem
        masked_img = real_img * (1.0 - mask)

        return masked_img, mask, real_img, 0