# train.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from preprocessing.pipeline import PreprocessingPipeline
from model.network          import SpamDetectorNN
from training.dataset       import EmailDataset
from training.trainer       import SpamTrainer
from training.evaluate      import ModelEvaluator

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    # Preprocessing
    "max_features"   : 10000,   # TF-IDF vocabulary size

    # Model
    "hidden_sizes"   : [512, 256, 128],
    "dropout_rate"   : 0.3,

    # Training
    "batch_size"     : 256,
    "learning_rate"  : 1e-3,
    "weight_decay"   : 1e-4,
    "epochs"         : 30,
    "patience"       : 7,       # early stopping

    # Paths
    "train_path"     : "data/processed/train.csv",
    "val_path"       : "data/processed/val.csv",
    "test_path"      : "data/processed/test.csv",
    "model_path"     : "model/saved/spam_model.pt",
    "pipeline_path"  : "model/saved/preprocessor.pkl",
}

def main():
    print("=" * 65)
    print("   NEURAL SPAM DETECTOR - TRAINING PIPELINE")
    print("=" * 65)

    # ── Step 1: Load CSVs ─────────────────────────────────────────────
    print("\n[1/6] Loading datasets...")
    train_df = pd.read_csv(CONFIG["train_path"])
    val_df   = pd.read_csv(CONFIG["val_path"])
    test_df  = pd.read_csv(CONFIG["test_path"])

    print(f"  Train : {len(train_df):,} emails")
    print(f"  Val   : {len(val_df):,} emails")
    print(f"  Test  : {len(test_df):,} emails")

    # ── Step 2: Preprocessing ─────────────────────────────────────────
    print("\n[2/6] Building preprocessing pipeline...")
    pipeline = PreprocessingPipeline(max_features=CONFIG["max_features"])

    # FIT only on training data - NEVER on val/test
    print("  Fitting on training data...")
    X_train = pipeline.fit_transform(train_df["text"].tolist())

    # TRANSFORM val and test using the FITTED pipeline
    print("  Transforming val data...")
    X_val   = pipeline.transform(val_df["text"].tolist())

    print("  Transforming test data...")
    X_test  = pipeline.transform(test_df["text"].tolist())

    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    print(f"\n  Feature matrix shapes:")
    print(f"    Train : {X_train.shape}")
    print(f"    Val   : {X_val.shape}")
    print(f"    Test  : {X_test.shape}")

    # Save pipeline immediately after fitting
    print("\n  Saving preprocessing pipeline...")
    pipeline.save(CONFIG["pipeline_path"])

    # ── Step 3: DataLoaders ───────────────────────────────────────────
    print("\n[3/6] Creating DataLoaders...")
    train_dataset = EmailDataset(X_train, y_train)
    val_dataset   = EmailDataset(X_val,   y_val)
    test_dataset  = EmailDataset(X_test,  y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = CONFIG["batch_size"],
        shuffle     = True,     # shuffle each epoch
        num_workers = 0,        # 0 = main process (safe on Windows)
        pin_memory  = False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = CONFIG["batch_size"],
        shuffle     = False,
        num_workers = 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = CONFIG["batch_size"],
        shuffle     = False,
        num_workers = 0
    )

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    # ── Step 4: Build Model ───────────────────────────────────────────
    print("\n[4/6] Building model...")
    input_size = pipeline.total_feature_size   # 10010

    model = SpamDetectorNN(
        input_size   = input_size,
        hidden_sizes = CONFIG["hidden_sizes"],
        dropout_rate = CONFIG["dropout_rate"]
    )
    print(f"  Input size   : {input_size:,}")
    print(f"  Total params : {model.count_parameters():,}")

    # ── Step 5: Train ─────────────────────────────────────────────────
    print("\n[5/6] Starting training...")
    trainer = SpamTrainer(
        model         = model,
        learning_rate = CONFIG["learning_rate"],
        weight_decay  = CONFIG["weight_decay"]
    )

    # Handle class imbalance (auto-computes pos_weight)
    trainer.set_class_weights(y_train)

    # Run training loop
    history = trainer.fit(
        train_loader = train_loader,
        val_loader   = val_loader,
        epochs       = CONFIG["epochs"],
        patience     = CONFIG["patience"],
        save_path    = CONFIG["model_path"]
    )

    # ── Step 6: Evaluate on Test Set ──────────────────────────────────
    print("\n[6/6] Evaluating on test set...")

    # Load best saved model (not the last epoch - the BEST epoch)
    print("  Loading best checkpoint...")
    checkpoint = torch.load(CONFIG["model_path"], map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = ModelEvaluator(model)
    metrics   = evaluator.full_report(test_loader, split_name="Test")

    # ── Final Summary ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"  Best epoch     : {history['best_epoch']}")
    print(f"  Best val loss  : {history['best_val_loss']:.4f}")
    print(f"  Test F1-Score  : {metrics['f1']:.4f}")
    print(f"  Test ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n  Saved files:")
    print(f"    Model    -> {CONFIG['model_path']}")
    print(f"    Pipeline -> {CONFIG['pipeline_path']}")
    print(f"\n  Next steps:")
    print(f"    API      -> uvicorn api.main:app --reload")
    print(f"    Gmail    -> python automation/run_pipeline.py")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()