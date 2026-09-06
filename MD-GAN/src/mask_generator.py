import torch
import numpy as np
import cv2
import random

class FreeFormMaskGenerator:

    def __init__(self, img_size=256, max_parts=8, max_vertex=12, max_length=100, max_brush_width=24):
        self.img_size = img_size
        self.max_parts = max_parts
        self.max_vertex = max_vertex
        self.max_length = max_length
        self.max_brush_width = max_brush_width

    def __call__(self, max_coverage=0.5):
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        num_parts = random.randint(1, self.max_parts)

        for _ in range(num_parts):
            num_vertex = random.randint(1, self.max_vertex)
            start_x = random.randint(0, self.img_size - 1)
            start_y = random.randint(0, self.img_size - 1)

            for _ in range(num_vertex):
                angle = random.uniform(0, 2 * np.pi)
                length = random.randint(10, self.max_length)
                brush_width = random.randint(8, self.max_brush_width)

                end_x = int(start_x + length * np.cos(angle))
                end_y = int(start_y + length * np.sin(angle))

                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_width)
                cv2.circle(mask, (start_x, start_y), brush_width // 2, 1.0, -1)
                cv2.circle(mask, (end_x, end_y), brush_width // 2, 1.0, -1)

                start_x, start_y = end_x, end_y

            if (mask.sum() / (self.img_size * self.img_size)) > max_coverage:
                break

        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return mask