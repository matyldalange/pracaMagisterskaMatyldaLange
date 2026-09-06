import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np


class MRIDataset(Dataset):
    def __init__(self, root_dir, img_size=256):
        self.root_dir = root_dir
        self.image_paths = []

        if os.path.exists(root_dir):
            for label in os.listdir(root_dir):
                path = os.path.join(root_dir, label)
                if os.path.isdir(path):
                    for f in os.listdir(path):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                            self.image_paths.append(os.path.join(path, f))

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            # normalizacja do zakresu [-1, 1]
            T.Normalize((0.5,), (0.5,))
        ])

    # generowanie losowych masek prostokątnych
    def create_random_mask(self, size):
        mask = torch.ones((1, size, size))
        y, x = np.random.randint(40, size - 80, 2)
        h, w = np.random.randint(30, 70, 2)
        mask[:, y:y + h, x:x + w] = 0
        return mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        label_name = os.path.basename(os.path.dirname(img_path)).lower()
        mapping = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
        label = mapping.get(label_name, 0)

        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        mask = self.create_random_mask(img_tensor.shape[1])

        # wypełnienie obszaru maski wartością -1.0, czyli czarnym pikselem w normalizacji
        masked_img = img_tensor.clone()
        masked_img[mask.expand_as(img_tensor) == 0] = -1.0

        return masked_img, mask, img_tensor, label

    def __len__(self):
        return len(self.image_paths)