# preprocessing/pipeline.py

# preprocessing/pipeline.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from pathlib import Path
from typing import List

from preprocessing.cleaner   import EmailCleaner
from preprocessing.vectorizer import TFIDFVectorizer
from preprocessing.features  import MetadataFeatureExtractor


class PreprocessingPipeline:
    """
    Single object = entire preprocessing workflow.

    TRAINING:   pipeline.fit_transform(train_emails)
    INFERENCE:  pipeline.transform_single(gmail_email)  ← IDENTICAL steps

    WHY ONE OBJECT?
    We pickle this entire object. When we load it during Gmail inference,
    we get the EXACT same vocabulary, EXACT same IDF weights, EXACT same
    cleaning rules. No drift. No bugs. Guaranteed consistency.
    """

    def __init__(self, max_features: int = 10000):
        self.cleaner        = EmailCleaner(remove_stopwords=True)
        self.vectorizer     = TFIDFVectorizer(max_features=max_features)
        self.meta_extractor = MetadataFeatureExtractor()
        self._is_fitted     = False

    # ── Size properties ───────────────────────────────────────────────
    @property
    def text_feature_size(self) -> int:
        return len(self.vectorizer.vocabulary_)

    @property
    def meta_feature_size(self) -> int:
        return self.meta_extractor.num_features

    @property
    def total_feature_size(self) -> int:
        return self.text_feature_size + self.meta_feature_size

    # ── Core methods ──────────────────────────────────────────────────
    def fit(self, raw_emails: List[str]) -> "PreprocessingPipeline":
        """Fit on TRAINING data only. Never call on val/test/Gmail."""
        print("  Step 1: Cleaning emails...")
        cleaned = [self.cleaner.clean(email) for email in raw_emails]

        print("  Step 2: Fitting TF-IDF vectorizer...")
        self.vectorizer.fit(cleaned)

        self._is_fitted = True
        print(f"  Total feature size: {self.total_feature_size:,} "
              f"(text: {self.text_feature_size:,} + meta: {self.meta_feature_size})")
        return self

    def transform(self, raw_emails: List[str]) -> np.ndarray:
        """Transform raw emails → feature matrix."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        # Text features — use CLEANED text
        cleaned       = [self.cleaner.clean(e) for e in raw_emails]
        text_features = self.vectorizer.transform(cleaned)

        # Metadata features — use RAW text (before cleaning!)
        meta_features = np.array(
            [self.meta_extractor.extract(e) for e in raw_emails],
            dtype=np.float32
        )

        # Concatenate: [TF-IDF (10000) | metadata (10)] = (10010,)
        combined = np.concatenate([text_features, meta_features], axis=1)
        return combined

    def fit_transform(self, raw_emails: List[str]) -> np.ndarray:
        return self.fit(raw_emails).transform(raw_emails)

    def transform_single(self, raw_email: str) -> np.ndarray:
        """Transform ONE email. Used during real-time Gmail inference."""
        return self.transform([raw_email])[0]

    # ── Save / Load ───────────────────────────────────────────────────
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"  [OK] Pipeline saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "PreprocessingPipeline":
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"  [OK] Pipeline loaded <- {path}")
        return pipeline


# ── Test the full pipeline ────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    print("="*55)
    print("Testing Preprocessing Pipeline")
    print("="*55)

    # Load a sample of training data
    train_df = pd.read_csv("data/processed/train.csv")
    sample   = train_df.head(500)  # Use 500 for quick test

    print(f"\nUsing {len(sample)} emails for quick test...\n")

    # Build pipeline
    pipeline = PreprocessingPipeline(max_features=1000)  # Small for testing

    # Fit + transform
    X = pipeline.fit_transform(sample['text'].tolist())
    y = sample['label'].values

    print(f"\nFeature matrix shape : {X.shape}")
    print(f"Labels shape         : {y.shape}")
    print(f"Feature min/max      : {X.min():.4f} / {X.max():.4f}")
    print(f"Non-zero features    : {(X > 0).sum():,}")

    # Test single email inference
    print("\n--- Single Email Inference Test ---")
    test_spam = "CONGRATULATIONS!!! You WON $1,000,000! Click NOW: http://win.com"
    test_ham  = "Hi, please review the attached report before tomorrow's meeting."

    vec_spam = pipeline.transform_single(test_spam)
    vec_ham  = pipeline.transform_single(test_ham)

    print(f"Spam email → vector shape: {vec_spam.shape}, non-zeros: {(vec_spam > 0).sum()}")
    print(f"Ham email  → vector shape: {vec_ham.shape},  non-zeros: {(vec_ham > 0).sum()}")

    # Save and reload test
    print("\n--- Save/Load Test ---")
    pipeline.save("model/saved/preprocessor_test.pkl")
    loaded = PreprocessingPipeline.load("model/saved/preprocessor_test.pkl")

    vec_reload = loaded.transform_single(test_spam)
    match = np.allclose(vec_spam, vec_reload)
    print(f"Save/reload vectors match: {'✅ YES' if match else '❌ NO'}")

    print("\n✅ Preprocessing pipeline working correctly!")
    print("🚀 Next step: python training/dataset.py")