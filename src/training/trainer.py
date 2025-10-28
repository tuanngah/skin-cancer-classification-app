import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import time
import numpy as np

from src.utils.focal_loss import FocalLoss
from src.utils.metrics import calculate_metrics, plot_confusion_matrix

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        self.criterion = FocalLoss(gamma=config['TRAIN']['FOCAL_LOSS_GAMMA']).to(device)

        self.patience = config['TRAIN']['EARLY_STOPPING_PATIENCE']
        self.best_metric = -1.0
        self.epochs_no_improve = 0
        self.stop_training = False

        self.mixup_prob = 0.0
        self.cutmix_prob = 0.0
        self.warmup_epochs = config['TRAIN']['WARMUP_EPOCHS']
        self.base_lr = config['TRAIN']['LEARNING_RATE']

        if train_loader:
            self.len_dataloader = len(train_loader)
        else:
            self.len_dataloader = 0

        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_macro_f1': [],
            'val_accuracy': []
        }

    def _apply_mix_or_cut(self, inputs, targets):
        return inputs, targets, 'none'

    def _adjust_lr_for_warmup(self, epoch, total_steps):
        if epoch < self.warmup_epochs:
            if self.len_dataloader == 0:
                return None

            current_step = total_steps % self.len_dataloader
            progress = (epoch * self.len_dataloader + current_step) / (self.warmup_epochs * self.len_dataloader)
            new_lr = self.base_lr * progress

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            return new_lr
        return None

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        # Handle case where train_loader might be None (though unlikely if run() is called)
        if not self.train_loader:
            print("Warning: train_loader is None in train_epoch. Skipping training epoch.")
            return 0.0


        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for step, (inputs, targets) in enumerate(pbar):
            # Skip batches with invalid labels (e.g., -1 from dataset error handling)
            valid_indices = targets != -1
            if not valid_indices.all():
                 print(f"Warning: Skipping batch {step} due to invalid labels.")
                 inputs = inputs[valid_indices]
                 targets = targets[valid_indices]
                 if inputs.nelement() == 0: # Check if batch became empty
                     continue # Skip empty batch


            inputs, targets = inputs.to(self.device), targets.to(self.device)

            current_lr = self._adjust_lr_for_warmup(epoch, step)
            inputs, targets, mix_type = self._apply_mix_or_cut(inputs, targets)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            current_lr_display = current_lr if current_lr else self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{current_lr_display:.6f}'})

        # Avoid division by zero if train_loader is empty
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0


        if self.config['TRAIN']['USE_SCHEDULER'] and epoch >= self.warmup_epochs and self.scheduler:
            self.scheduler.step()

        return avg_loss

    @torch.no_grad()
    def validate_epoch(self, loader):
        # Handle case where validation loader might be None
        if not loader:
             print("Warning: loader is None in validate_epoch. Skipping validation.")
             # Return default/dummy values
             return 0.0, {'accuracy': 0.0, 'macro_f1': 0.0, 'auc_macro_ovr': 0.0, 'confusion_matrix': np.zeros((self.config['NUM_CLASSES'], self.config['NUM_CLASSES'])), 'y_pred_probs': np.array([])}


        self.model.eval()
        total_loss = 0
        all_targets = []
        all_outputs = []

        pbar = tqdm(loader, desc="[Validate]", leave=False)
        for inputs, targets in pbar:
             # Skip batches with invalid labels
             valid_indices = targets != -1
             if not valid_indices.all():
                 print(f"Warning: Skipping validation batch due to invalid labels.")
                 inputs = inputs[valid_indices]
                 targets = targets[valid_indices]
                 if inputs.nelement() == 0:
                     continue


             inputs, targets = inputs.to(self.device), targets.to(self.device)
             outputs = self.model(inputs)
             # Ensure loss calculation handles potentially empty batches if all were invalid
             if inputs.nelement() > 0:
                 loss = self.criterion(outputs, targets)
                 total_loss += loss.item()
                 all_targets.append(targets)
                 all_outputs.append(outputs)
             else:
                 # Handle cases where the filtered batch is empty
                 pass # No loss contribution, no targets/outputs


        # Avoid division by zero if loader is empty or all batches were skipped
        if len(loader) == 0 or not all_outputs:
             avg_loss = 0.0
             # Return default/dummy metrics if no valid outputs were collected
             metrics = {'accuracy': 0.0, 'macro_f1': 0.0, 'auc_macro_ovr': 0.0, 'confusion_matrix': np.zeros((self.config['NUM_CLASSES'], self.config['NUM_CLASSES'])), 'y_pred_probs': np.array([])}
        else:
             avg_loss = total_loss / len(loader)
             all_targets = torch.cat(all_targets)
             all_outputs = torch.cat(all_outputs)
             metrics = calculate_metrics(all_targets, all_outputs, self.config['NUM_CLASSES'])


        return avg_loss, metrics

    def run(self):
        print(f"Starting training for {self.config['TRAIN']['EPOCHS']} epochs.")

        self._load_checkpoint()

        # Check if loaders are valid before starting the loop
        if not self.train_loader or not self.val_loader:
            print("Error: train_loader or val_loader is None. Cannot start training.")
            # Return dummy history if training cannot start
            return self.model, self.history


        for epoch in range(1, self.config['TRAIN']['EPOCHS'] + 1):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.validate_epoch(self.val_loader)
            elapsed_time = time.time() - start_time

            print(f"Epoch {epoch} | Time: {elapsed_time:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro-F1: {metrics['macro_f1']:.4f} (Best: {self.best_metric:.4f})")
            print(f"  -> F1 (mel): {metrics.get('f1_mel', 0):.4f} | Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_macro_f1'].append(metrics['macro_f1'])
            self.history['val_accuracy'].append(metrics['accuracy'])

            self._check_early_stopping(metrics['macro_f1'])

            if self.stop_training:
                break

        print("\n--- Training finished. Loading best model ---")
        self._load_checkpoint(is_best=True)

        return self.model, self.history

    def _check_early_stopping(self, current_metric):
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.epochs_no_improve = 0
            self._save_checkpoint(is_best=True)
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                print(f"\n--- Early Stopping triggered after {self.patience} epochs without improvement. ---")
                self.stop_training = True

    def _save_checkpoint(self, is_best=False):
        state = {
            'epoch': self.epochs_no_improve,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_macro_f1': self.best_metric
        }

        checkpoint_dir = os.path.join('models', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        filepath = os.path.join(checkpoint_dir, f"{self.config['RUN_NAME']}_last.pth")
        torch.save(state, filepath)

        if is_best:
            best_filepath = os.path.join(checkpoint_dir, f"{self.config['RUN_NAME']}_best.pth")
            torch.save(state, best_filepath)

    def _load_checkpoint(self, is_best=False):
        checkpoint_dir = os.path.join('models', 'checkpoints')

        if is_best:
            filepath = os.path.join(checkpoint_dir, f"{self.config['RUN_NAME']}_best.pth")
        else:
            filepath = os.path.join(checkpoint_dir, f"{self.config['RUN_NAME']}_last.pth")

        if os.path.exists(filepath):
            print(f"Loading checkpoint from: {filepath}")
            try:
                 # Load with weights_only=True for security if checkpoint is trusted source or format is known
                 checkpoint = torch.load(filepath, map_location=self.device, weights_only=False) # Set to True if possible
                 self.model.load_state_dict(checkpoint['model_state_dict'])
                 self.best_metric = checkpoint['best_macro_f1']

                 if not is_best and self.optimizer and 'optimizer_state_dict' in checkpoint:
                     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 elif not is_best and self.optimizer:
                     print("Warning: Checkpoint loaded but optimizer state not found or optimizer not initialized.")

            except Exception as e:
                 print(f"Error loading checkpoint: {e}. Starting fresh.")
                 self.best_metric = -1.0 # Reset best metric if loading fails
        # else:
            # print("No checkpoint found. Starting training from scratch.") # Optional message