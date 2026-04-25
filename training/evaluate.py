# training/evaluate.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score
)

class ModelEvaluator:

    def __init__(self, model, device: str = None):
        self.model  = model
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def get_predictions(self, loader: DataLoader):
        self.model.eval()
        all_labels, all_probs = [], []

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                probs    = self.model.predict_proba(features).squeeze(1)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        labels = np.array(all_labels)
        probs  = np.array(all_probs)
        preds  = (probs >= 0.5).astype(int)
        return labels, preds, probs

    def full_report(self, loader: DataLoader, split_name: str = "Test"):
        labels, preds, probs = self.get_predictions(loader)

        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

        accuracy  = (tp + tn) / len(labels)
        precision = precision_score(labels, preds, zero_division=0)
        recall    = recall_score(labels, preds, zero_division=0)
        f1        = f1_score(labels, preds, zero_division=0)
        roc_auc   = roc_auc_score(labels, probs)
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"\n{'='*55}")
        print(f"  Evaluation Report - {split_name} Set")
        print(f"{'='*55}")
        print(f"  Accuracy       : {accuracy:.4f}")
        print(f"  Precision      : {precision:.4f}  "
              f"<- of predicted spam, how many ARE spam?")
        print(f"  Recall         : {recall:.4f}  "
              f"<- of actual spam, how many did we catch?")
        print(f"  F1-Score       : {f1:.4f}  <- main metric")
        print(f"  ROC-AUC        : {roc_auc:.4f}  <- 1.0 is perfect")
        print(f"  False Pos Rate : {fpr:.4f}  "
              f"<- ham wrongly flagged as spam")

        print(f"\n  Confusion Matrix:")
        print(f"  {'':20} Predicted Ham  Predicted Spam")
        print(f"  {'Actual Ham':20} {tn:>13,}  {fp:>13,}")
        print(f"  {'Actual Spam':20} {fn:>13,}  {tp:>13,}")

        dummy_acc = 1 - labels.mean()
        print(f"\n  Dummy classifier accuracy : {dummy_acc:.4f}")
        print(f"  Your model accuracy       : {accuracy:.4f}")
        print(f"  Improvement               : "
              f"+{(accuracy - dummy_acc)*100:.2f}%")

        return {
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1,
            "roc_auc": roc_auc, "fpr": fpr
        }