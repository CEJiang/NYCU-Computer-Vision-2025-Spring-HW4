"""Dataset class for image restoration with optional TTA augmentation."""

import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class RestorationDataset(Dataset):
    """Custom Dataset for image restoration tasks (e.g., snow/rain removal)."""

    def __init__(self, root_dir, filenames=None, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        degraded_dir = os.path.join(self.root_dir, "degraded")
        clean_dir = os.path.join(self.root_dir, "clean")

        self.filenames = sorted(filenames) if filenames else sorted(
            os.listdir(degraded_dir))

        self.degraded_paths = [
            os.path.join(
                degraded_dir,
                f) for f in self.filenames]
        self.clean_paths = [
            os.path.join(clean_dir,
                         f.replace(".png", "").replace("snow-", "snow_clean-").replace("rain-", "rain_clean-") + ".png")
            for f in self.filenames
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        degraded_image = Image.open(self.degraded_paths[index]).convert("RGB")
        clean_image = Image.open(self.clean_paths[index]).convert("RGB")

        # TTA augmentation variants
        tta_variants = [
            lambda x: x,                         # 1. original
            F.hflip,                             # 2. horizontal flip
            F.vflip,                             # 3. vertical flip
            lambda x: F.vflip(F.hflip(x)),
            # 4. horizontal + vertical flip
            lambda x: F.rotate(x, 180),          # 5. rotate 180°
            lambda x: F.hflip(F.rotate(x, 180)),  # 6. rotate 180° + hflip
            lambda x: F.vflip(F.rotate(x, 180)),  # 7. rotate 180° + vflip
            lambda x: F.vflip(F.hflip(F.rotate(x, 180)))  # 8. all
        ]

        aug_fn = random.choice(tta_variants)
        degraded_image = aug_fn(degraded_image)
        clean_image = aug_fn(clean_image)

        if self.transforms:
            degraded_image = self.transforms(degraded_image)
            clean_image = self.transforms(clean_image)

        task_type = "snow" if "snow" in self.filenames[index].lower(
        ) else "rain"
        return degraded_image, clean_image, task_type
