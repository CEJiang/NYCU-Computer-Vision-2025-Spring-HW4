"""Test pipeline for image restoration using PromptIR model with TTA and optional photometric augmentations."""

import os
import zipfile
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


def get_transforms():
    """Return default test-time transformation."""
    return T.Compose([T.ToTensor()])


def collate_fn(batch):
    """Collate function for batching images and filenames."""
    imgs, fnames = zip(*batch)
    return torch.stack(imgs), fnames


class TestDataset(Dataset):
    """Dataset for test images from degraded folder."""

    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        degraded_dir = os.path.join(self.root_dir, "degraded")
        self.filenames = sorted([
            f for f in os.listdir(degraded_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.degraded_paths = [
            os.path.join(
                degraded_dir,
                f) for f in self.filenames]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        degraded_image = Image.open(self.degraded_paths[index]).convert("RGB")
        if self.transforms:
            degraded_image = self.transforms(degraded_image)
        return degraded_image, self.filenames[index]


def load_data(data_path, batch_size):
    """Load test dataset and return DataLoader."""
    dataset = TestDataset(
        root_dir=os.path.join(data_path, "test"),
        transforms=get_transforms()
    )
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False, collate_fn=collate_fn)


def tta_forward(model, image):
    """Perform test-time augmentation (TTA) on a single image and average predictions."""
    variants = [
        (lambda x: x, lambda x: x),
        (lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
        (lambda x: torch.flip(x, dims=[1]), lambda x: torch.flip(x, dims=[1])),
        (lambda x: torch.flip(torch.flip(x, dims=[1]), dims=[2]),
         lambda x: torch.flip(torch.flip(x, dims=[1]), dims=[2])),
        (lambda x: torch.rot90(x, 2, dims=[1, 2]),
         lambda x: torch.rot90(x, -2, dims=[1, 2])),
        (lambda x: torch.rot90(torch.flip(x, dims=[2]), 2, dims=[1, 2]),
         lambda x: torch.flip(torch.rot90(x, -2, dims=[1, 2]), dims=[2])),
        (lambda x: torch.rot90(torch.flip(x, dims=[1]), 2, dims=[1, 2]),
         lambda x: torch.flip(torch.rot90(x, -2, dims=[1, 2]), dims=[1])),
        (lambda x: torch.rot90(torch.flip(torch.flip(x, dims=[1]), dims=[2]), 2, dims=[1, 2]),
         lambda x: torch.flip(torch.flip(torch.rot90(x, -2, dims=[1, 2]), dims=[1]), dims=[2])),
    ]

    preds = []
    for aug_fn, reverse_fn in variants:
        aug_img = aug_fn(image)
        pred = model(aug_img.unsqueeze(0))
        pred = reverse_fn(pred.squeeze(0)).unsqueeze(0)
        preds.append(pred)

    return torch.stack(preds).mean(dim=0).clamp(0, 1)


@torch.no_grad()
def test_model(device, model, args, output_npz_path="pred.npz"):
    """Run model inference on test set and save results as .npz and .zip."""
    model_path = os.path.join(args.saved_path, "best_model_45.pth")
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")

    test_loader = load_data(args.data_path, args.batch_size)
    results = {}

    for degraded_imgs, filenames in tqdm(test_loader):
        degraded_imgs = [img.to(device) for img in degraded_imgs]

        for img_tensor, fname in zip(degraded_imgs, filenames):
            pred = tta_forward(model, img_tensor)
            pred_np = pred.clamp(0, 1).squeeze(0).cpu().numpy()
            pred_np = (pred_np * 255).astype(np.uint8)
            results[fname] = pred_np

    np.savez(output_npz_path, **results)
    print(f"Saved {len(results)} results to {output_npz_path}")

    zip_path = output_npz_path.replace(".npz", ".zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_npz_path, arcname=os.path.basename(output_npz_path))

    print(f"Zipped result to {zip_path}")
