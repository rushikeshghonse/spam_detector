# training/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from typing import Dict, List, Tuple


class SpamTrainer:

    def __init__(self,
                 model:         nn.Module,
                 learning_rate: float = 1e-3,
                 weight_decay:  float = 1e-4,
                 device:        str   = None):

        self.model  = model
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        print(f"  Training device: {self.device}")

        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr           = learning_rate,
            weight_decay = weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode     = 'min',
            factor   = 0.5,
            patience = 3,
        )

        self.train_losses: List[float] = []
        self.val_losses:   List[float] = []
        self.best_val_loss = float('inf')
        self.best_epoch    = 0

    def set_class_weights(self, labels: np.ndarray):
        n_ham  = (labels == 0).sum()
        n_spam = (labels == 1).sum()
        pos_weight = torch.tensor(
            [n_ham / n_spam], dtype=torch.float32
        ).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"  Class weights -> ham: {n_ham:,}, "
              f"spam: {n_spam:,}, pos_weight: {pos_weight.item():.3f}")

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = len(loader)

        for batch_idx, (features, labels) in enumerate(loader):
            features = features.to(self.device)
            labels   = labels.to(self.device).unsqueeze(1)

            logits = self.model(features)
            loss   = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                print(f"    Batch {batch_idx+1:>3}/{n_batches} | "
                      f"Loss: {loss.item():.4f}", end="\r")

        print()
        return total_loss / n_batches

    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels   = labels.to(self.device).unsqueeze(1)

                logits = self.model(features)
                loss   = self.criterion(logits, labels)
                preds  = (torch.sigmoid(logits) >= 0.5).float()

                total_loss += loss.item()
                correct    += (preds == labels).sum().item()
                total      += labels.size(0)

        return total_loss / len(loader), correct / total

    def fit(self,
            train_loader: DataLoader,
            val_loader:   DataLoader,
            epochs:       int = 30,
            patience:     int = 7,
            save_path:    str = "model/saved/spam_model.pt") -> Dict:

        print(f"\n{'='*65}")
        print(f"  Epochs: {epochs}  |  Early stop patience: {patience}")
        print(f"{'='*65}")
        print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
              f"{'Val Acc':>8} | {'LR':>10} | {'Status':<10}")
        print(f"  {'-'*65}")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss        = self._train_one_epoch(train_loader)
            val_loss, val_acc = self._evaluate(val_loader)
            elapsed           = time.time() - t0
            current_lr        = self.optimizer.param_groups[0]['lr']

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch    = epoch
                patience_counter   = 0
                self._save_checkpoint(save_path)
                status = "[saved]"
            else:
                patience_counter += 1
                status = f"[wait {patience_counter}/{patience}]"

            print(f"  {epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
                  f"{val_acc:>7.2%} | {current_lr:>10.6f} | {status}")

            self.scheduler.step(val_loss)

            if patience_counter >= patience:
                print(f"\n  Early stopping triggered at epoch {epoch}.")
                break

        print(f"\n  Best -> Epoch {self.best_epoch}, "
              f"Val Loss: {self.best_val_loss:.4f}")
        print(f"  Model saved to: {save_path}")

        return {
            "train_losses" : self.train_losses,
            "val_losses"   : self.val_losses,
            "best_epoch"   : self.best_epoch,
            "best_val_loss": self.best_val_loss
        }

    def _save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size'  : self.model.input_size,
                'hidden_sizes': self.model.hidden_sizes,
                'dropout_rate': self.model.dropout_rate,
            },
            'best_val_loss': self.best_val_loss,
        }, path)