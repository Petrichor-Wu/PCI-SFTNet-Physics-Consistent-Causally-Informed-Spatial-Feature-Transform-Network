import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.amp
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Import your data loader and model
from data_loader import CausalDownscalingDataset, calculate_and_save_stats, STATS_FILE
from pci_sftnet import PCI_SFTNet

# Set matplotlib backend to prevent headless errors
matplotlib.use('Agg')

# --- Configuration ---
PCMCI_JSON_PATH = r'E:\graduation project\coding\PCI-SFTNet\Causality_tree\PCMCI_Nonlinear_Analysis_Output\consensus_structure\consensus_pcmci_tree_NAMES.txt'
CHECKPOINT_DIR = './checkpoints'

# --- Performance optimization hyperparameters ---
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50
GPU_ID = 0
NUM_WORKERS = 4
TRAIN_RATIO = 0.9


class PureL1Loss(nn.Module):
    """
    Retain only L1 Loss (Masked) for basic pixel-level approximation
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, pred, target, boundary_mask):
        # Calculate pixel error
        loss_map = self.l1(pred, target)
        # Apply boundary mask (only calculate loss in valid areas)
        loss = (loss_map * boundary_mask).sum() / (boundary_mask.sum() + 1e-6)
        return loss


class LossVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.train_losses = []
        self.val_losses = []

    def update(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def plot(self):
        """
        🔥 [Modified] Only plot one graph: Train Loss vs Val Loss
        """
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(12, 8))  # Slightly enlarge canvas to prevent text crowding

        # Plot training loss curve
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=3)  # Thicken the line slightly
        # Plot validation loss curve
        plt.plot(epochs, self.val_losses, 'r--', label='Validation Loss', linewidth=3)

        # Set title and axis label font
        plt.title('Training and Validation Loss', fontsize=25, fontweight='bold')
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss (L1)', fontsize=20)

        # 🔥 [Key modification] Set tick label font size
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # Set legend font size
        plt.legend(loc='best', fontsize=16)

        plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines

        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'loss.png')
        plt.savefig(save_path, dpi=300)  # Increase resolution
        plt.close()


def train():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    if not os.path.exists(STATS_FILE):
        print("Data statistics file not found, starting automatic calculation...")
        calculate_and_save_stats()
    else:
        print(f"✅ Existing statistics file detected: {STATS_FILE}")

    visualizer = LossVisualizer(CHECKPOINT_DIR)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{GPU_ID}')
        torch.backends.cudnn.benchmark = True
        print(f"✅ GPU Mode: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("❌ CPU Mode")

    scaler = torch.amp.GradScaler('cuda')

    FULL_START, FULL_END = '2016-01-01', '2016-12-31'
    print(f"Building complete dataset ({FULL_START} ~ {FULL_END})...")

    full_dataset = CausalDownscalingDataset(
        start_date=FULL_START, end_date=FULL_END, patch_size=128, is_train=True
    )

    if len(full_dataset) == 0:
        print("Error: Dataset is empty.")
        return

    total_size = len(full_dataset)
    train_size = int(total_size * TRAIN_RATIO)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    print(f"Dataset randomly split: Train {len(train_dataset)}, Val {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = PCI_SFTNet(pcmci_json_path=PCMCI_JSON_PATH).to(device)

    if torch.cuda.device_count() > 1:
        print(f"🚀 Detected {torch.cuda.device_count()} GPUs! Activating DataParallel...")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    criterion = PureL1Loss().to(device)

    print("Starting training (Pure L1 Loss)...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        valid_steps = 0
        epoch_start_time = time.time()

        old_lr = optimizer.param_groups[0]['lr']

        for i, batch in enumerate(train_loader):
            t0 = time.time()
            gpu_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            if 'Label' not in gpu_batch: continue

            target = gpu_batch['Label']
            boundary_mask = gpu_batch.get('BoundaryMask', torch.ones_like(target))

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                pred = model(gpu_batch)
                loss = criterion(pred, target, boundary_mask)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            t1 = time.time()
            step_time = t1 - t0

            epoch_loss += loss.item()
            valid_steps += 1

            if i % 10 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS} [Train] | Step {i}/{len(train_loader)} | "
                      f"Time: {step_time:.3f}s | Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / max(valid_steps, 1)

        # --- Validation Phase ---
        model.eval()
        val_loss_sum = 0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                gpu_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if
                             isinstance(v, torch.Tensor)}

                if 'Label' not in gpu_batch: continue

                target = gpu_batch['Label']
                boundary_mask = gpu_batch.get('BoundaryMask', torch.ones_like(target))

                with torch.amp.autocast('cuda'):
                    pred = model(gpu_batch)
                    loss = criterion(pred, target, boundary_mask)

                val_loss_sum += loss.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        epoch_duration = time.time() - epoch_start_time

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if current_lr < old_lr:
            print(f"📉 Learning Rate Reduced: {old_lr:.2e} -> {current_lr:.2e}")

        print(f"Epoch {epoch + 1} Done. LR: {current_lr:.2e} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print("-" * 30)

        # Update and plot
        visualizer.update(avg_train_loss, avg_val_loss)
        visualizer.plot()

        save_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"🌟 New Best Model Saved! (Val Loss: {best_val_loss:.4f})")


if __name__ == '__main__':
    train()