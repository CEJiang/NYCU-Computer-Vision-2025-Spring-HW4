"""Training and validation pipeline for PromptIR image restoration model."""

import os
import time
import gc
import random
import torch
import torch.nn.functional as F
from piq import ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms as T

from dataset import RestorationDataset
from utils import plot_loss_accuracy, plot_psnr_curve


def denormalize(tensor):
    """Undo normalization from [-1, 1] to [0, 1]."""
    return tensor * 0.5 + 0.5


def get_transforms():
    """Return image transformation pipeline."""
    return T.Compose([T.ToTensor()])


def collate_fn(batch):
    """Collate function for image and label batching."""
    degraded_imgs, clean_imgs, task_types = zip(*batch)
    return list(degraded_imgs), list(clean_imgs), list(task_types)


def load_data(data_path, args):
    """Load training and validation datasets."""
    all_filenames = sorted(os.listdir(os.path.join(data_path, "degraded")))
    random.seed(42)
    random.shuffle(all_filenames)

    split_ratio = 0.8
    split_idx = int(split_ratio * len(all_filenames))
    train_files = all_filenames[:split_idx]
    valid_files = all_filenames[split_idx:]

    train_dataset = RestorationDataset(
        root_dir=data_path,
        filenames=train_files,
        transforms=get_transforms())

    valid_dataset = RestorationDataset(
        root_dir=data_path,
        filenames=valid_files,
        transforms=get_transforms())

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn)

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn)

    return train_loader, valid_loader


class Trainer:
    """Trainer class for handling model training, validation, and saving."""

    def __init__(self, device, model, optimizer, scheduler, args):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float("inf")
        self.pixel_loss_l1 = torch.nn.L1Loss()
        self.pixel_loss_l2 = torch.nn.MSELoss()
        self.psnrs = []
        self.best_psnr = -float("inf")
        self.ratio = 0.3

    def train(self, train_loader, epoch):
        """Training loop for one epoch."""
        self.model.train()
        total_loss = 0.0

        if (epoch + 1) >= 20 and (epoch + 1) % 5 == 0:
            self.ratio = min(self.ratio + 0.05, 0.7)

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1} [Train]"
        )
        for _, (degraded_imgs, clean_imgs, _) in pbar:
            pbar.set_postfix({"Best PSNR": f"{self.best_psnr:.2f}"})
            degraded_imgs = torch.stack(degraded_imgs).to(self.device)
            clean_imgs = torch.stack(clean_imgs).to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(degraded_imgs)

            loss_l1 = self.pixel_loss_l1(preds, clean_imgs)
            loss_ssim = 1 - ssim(preds, clean_imgs, data_range=1.0)
            loss = (1 - self.ratio) * loss_l1 + self.ratio * loss_ssim

            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach().item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def validate(self, valid_loader, epoch):
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0

        pbar = tqdm(
            valid_loader,
            total=len(valid_loader),
            desc=f"Epoch {epoch+1} [Valid]")
        for degraded_imgs, clean_imgs, _ in pbar:
            degraded_imgs = torch.stack(degraded_imgs).to(self.device)
            clean_imgs = torch.stack(clean_imgs).to(self.device)

            preds = self.model(degraded_imgs)

            loss_l1 = self.pixel_loss_l1(preds, clean_imgs)
            loss_ssim = 1 - ssim(preds, clean_imgs, data_range=1.0)
            loss = (1 - self.ratio) * loss_l1 + self.ratio * loss_ssim
            total_loss += loss.item()

            mse = F.mse_loss(preds, clean_imgs, reduction='none')
            mse_per_img = mse.view(mse.size(0), -1).mean(dim=1)
            psnr_per_img = 20 * torch.log10(torch.tensor(1.0, device=self.device)) - \
                10 * torch.log10(mse_per_img)
            total_psnr += psnr_per_img.sum().item()

        avg_loss = total_loss / len(valid_loader)
        avg_psnr = total_psnr / len(valid_loader.dataset)

        print(
            f"[Epoch {epoch+1}] Val Loss: {avg_loss:.4f} | Avg PSNR: {avg_psnr:.2f} dB | best PSNR {self.best_psnr:.2f}")
        return avg_loss, avg_psnr

    def save_model(self, epoch, is_best=False):
        """Save latest and best model checkpoints."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
            'best_psnr': self.best_psnr,
            'psnrs': self.psnrs,
            'ratio': self.ratio
        }
        torch.save(
            checkpoint,
            os.path.join(
                self.args.saved_path,
                'latest_checkpoint.pth'))

        if is_best:
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.args.saved_path,
                    f"best_model_{epoch}.pth"))
            print("Best model is saved.")

    def load_model(self):
        """Resume training from last checkpoint."""
        path = os.path.join(self.args.saved_path, 'latest_checkpoint.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.best_loss = checkpoint['best_loss']
            self.best_psnr = checkpoint['best_psnr']
            self.psnrs = checkpoint.get('psnrs', [])
            self.ratio = checkpoint.get('ratio', 0.3)
            print(
                f"Resumed from epoch {checkpoint['epoch'] + 1} | best PSNR = {self.best_psnr:.2f}")
            return checkpoint['epoch'] + 1
        return 0


def train_model(device, model, optimizer, scheduler,
                train_loader, valid_loader, args):
    """Main training loop."""
    trainer = Trainer(device, model, optimizer, scheduler, args)
    start_epoch = trainer.load_model()

    for epoch in range(start_epoch, args.epochs):
        plot_loss_accuracy(trainer.train_losses, trainer.val_losses)
        plot_psnr_curve(trainer.psnrs)

        start = time.time()
        train_loss = trainer.train(train_loader, epoch)
        val_loss, psnr = trainer.validate(valid_loader, epoch)

        trainer.train_losses.append(train_loss)
        trainer.val_losses.append(val_loss)
        trainer.psnrs.append(psnr)

        is_best = psnr > trainer.best_psnr
        if is_best:
            trainer.best_psnr = psnr

        trainer.save_model(epoch, is_best=is_best)

        gc.collect()
        torch.cuda.empty_cache()
        print(f"Epoch {epoch + 1} time: {time.time() - start:.2f} sec")


def validate_model(device, model, valid_loader, args):
    """Standalone validation function (not implemented)."""
    raise NotImplementedError
