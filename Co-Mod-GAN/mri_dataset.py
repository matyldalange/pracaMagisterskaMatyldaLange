import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# wczytywanie obrazów oraz generowanie masek zajmujących od 10% do 30% powierzchni skanu MRI
class MRICoModDataset(Dataset):
    def __init__(self, root_dir, size=256, mode='train'):
        self.root_dir = root_dir
        self.size = size
        self.mode = mode
        self.paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir)
                      for f in filenames if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    # generowanie losowej maski prostokątnej zajmującej od 10% do 30% powierzchni skanu MRI
    def generate_medical_mask(self):
        mask = np.zeros((self.size, self.size), dtype=np.float32)

        total_area = self.size * self.size
        min_area = 0.10 * total_area
        max_area = 0.30 * total_area

        target_area = np.random.uniform(min_area, max_area)
        aspect_ratio = np.random.uniform(0.5, 2.0)

        w = int(np.sqrt(target_area * aspect_ratio))
        h = int(np.sqrt(target_area / aspect_ratio))

        w = min(w, self.size)
        h = min(h, self.size)

        x1 = np.random.randint(0, self.size - w + 1)
        y1 = np.random.randint(0, self.size - h + 1)

        mask[y1:y1 + h, x1:x1 + w] = 1.0

        return torch.from_numpy(mask).unsqueeze(0)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('L')
            img = self.transform(img)

            mask = self.generate_medical_mask()
            # nałożenie maski na obraz
            masked_img = img * (1 - mask)

            return {
                'gt': img,
                'masked': masked_img,
                'mask': mask
            }
        except Exception as e:
            print(f"Error loading {self.paths[idx]}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))