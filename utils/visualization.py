"""Visualization utilities for training curves such as loss, accuracy, and PSNR."""

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_accuracy(
    train_losses, val_losses,
    train_accuracies=None, val_accuracies=None,
    save_fig=True, output_path="training_curve.png"
):
    """
    Plot and optionally save training/validation loss and accuracy curves.

    Args:
        train_losses (list of float): Training loss per epoch.
        val_losses (list of float): Validation loss per epoch.
        train_accuracies (list of float, optional): Training accuracy per epoch.
        val_accuracies (list of float, optional): Validation accuracy per epoch.
        save_fig (bool): Whether to save the figure.
        output_path (str): Path to save the output plot.
    """
    num_plots = 2 if train_accuracies and val_accuracies else 1
    _, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    axes[0].plot(train_losses, label="Train Loss", marker='o')
    axes[0].plot(val_losses, label="Validation Loss", marker='o')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid()

    if num_plots == 2:
        axes[1].plot(train_accuracies, label="Train Acc", marker='o')
        axes[1].plot(val_accuracies, label="Val Acc", marker='o')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training & Validation Accuracy")
        axes[1].legend()
        axes[1].grid()

    plt.tight_layout()
    if save_fig:
        plt.savefig(output_path)
        print(f"Training curve saved as {output_path}")
    plt.close()


def plot_psnr_curve(psnrs, save_fig=True, output_path="psnr_curve.png"):
    """
    Plot and optionally save PSNR curve.

    Args:
        psnrs (list of float): PSNR values per epoch.
        save_fig (bool): Whether to save the figure.
        output_path (str): Path to save the output plot.
    """
    plt.figure(figsize=(7, 6))
    plt.plot(psnrs, label="PSNR", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR Curve")
    plt.legend()
    plt.grid()
    if save_fig:
        plt.savefig(output_path)
        print(f"PSNR curve saved as {output_path}")
    plt.close()
