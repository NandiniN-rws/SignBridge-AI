import argparse
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, f1_score
)
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.services.sign_model import SignCNNLSTMAttention


def make_splits(X, y, groups):
    """Use signer grouping only when IDs look like genuine cross-class signers."""
    unique_groups = np.unique(groups)
    group_labels = {
        g: set(y[groups == g].tolist()) for g in unique_groups
    }

    # A real signer should generally contribute samples to multiple classes.
    # The downloaded folder structure may instead encode one class-folder per group.
    valid_group_ids = (
        len(unique_groups) >= 3
        and not np.all(groups == "unknown")
        and sum(len(v) > 1 for v in group_labels.values()) >= 2
    )

    if valid_group_ids:
        try:
            outer = GroupShuffleSplit(n_splits=20, test_size=0.20, random_state=42)
            for trv, te in outer.split(X, y, groups):
                if len(np.unique(y[te])) == len(np.unique(y)):
                    inner = GroupShuffleSplit(n_splits=20, test_size=0.20, random_state=43)
                    for tr_rel, va_rel in inner.split(X[trv], y[trv], groups[trv]):
                        tr, va = trv[tr_rel], trv[va_rel]
                        if len(np.unique(y[tr])) == len(np.unique(y)) and len(np.unique(y[va])) == len(np.unique(y)):
                            print("Evaluation split: signer-grouped (all classes represented).")
                            return tr, va, te, True
        except ValueError:
            pass

    print("WARNING: signer IDs are unavailable or appear to be class-folder names.")
    print("Using stratified sample-level split for this provisional experiment.")
    print("Do NOT describe this result as signer-independent; participant leakage may remain.")
    indices = np.arange(len(y))
    trv, te = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=y
    )
    tr, va = train_test_split(
        trv, test_size=0.20, random_state=43, stratify=y[trv]
    )
    return tr, va, te, False


def main():
    parser = argparse.ArgumentParser(
        description="Train temporal CNN-BiLSTM-attention on ISL landmark sequences."
    )
    parser.add_argument("--data", default="deep_learning/data/landmarks_emergency.npz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="deep_learning/checkpoints/signbridge_video_cnn_lstm.pt")
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    labels = data["y"].astype(str)
    if X.ndim != 3 or X.shape[1:] != (32, 258):
        raise ValueError(f"Expected X shape (N,32,258), got {X.shape}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    groups = data["signer"].astype(str) if "signer" in data.files else np.array(["unknown"] * len(y))

    tr, va, te, signer_grouped = make_splits(X, y, groups)
    all_class_ids = np.arange(len(encoder.classes_))
    print(f"Samples: {len(y)} | Classes: {len(encoder.classes_)}")
    print(f"Split sizes: train={len(tr)}, validation={len(va)}, test={len(te)}")
    print("Train classes:", len(np.unique(y[tr])))
    print("Validation classes:", len(np.unique(y[va])))
    print("Test classes:", len(np.unique(y[te])))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training device:", device)
    model = SignCNNLSTMAttention(len(encoder.classes_)).to(device)

    present = np.unique(y[tr])
    class_weights = compute_class_weight("balanced", classes=present, y=y[tr])
    weights = torch.ones(len(encoder.classes_), device=device)
    for class_id, weight in zip(present, class_weights):
        weights[int(class_id)] = float(weight)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    def loader(ids, shuffle):
        dataset = TensorDataset(
            torch.tensor(X[ids], dtype=torch.float32),
            torch.tensor(y[ids], dtype=torch.long),
        )
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = loader(tr, True)
    val_loader = loader(va, False)
    test_loader = loader(te, False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    stale = 0

    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += loss.item() * len(xb)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_total += criterion(model(xb.to(device)), yb.to(device)).item() * len(xb)
        val_loss = val_total / max(len(va), 1)
        train_loss = train_total / max(len(tr), 1)
        scheduler.step(val_loss)
        print(
            f"epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": encoder.classes_.tolist(),
                    "architecture": "temporal-cnn-bilstm-attention",
                    "frames": 32,
                    "features": 258,
                    "evaluation_split": "signer_grouped" if signer_grouped else "stratified_sample_provisional",
                },
                output_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping.")
                break

    checkpoint = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    predictions = []
    actual = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            actual.extend(yb.numpy().tolist())

    print("\n===== TEST RESULTS =====")
    print("Test accuracy:", accuracy_score(actual, predictions))
    print("Macro F1:", f1_score(actual, predictions, labels=all_class_ids, average="macro", zero_division=0))
    print(
        classification_report(
            actual,
            predictions,
            labels=all_class_ids,
            target_names=encoder.classes_,
            zero_division=0,
        )
    )
    print("Confusion matrix:\n", confusion_matrix(actual, predictions, labels=all_class_ids))
    print("Checkpoint saved:", output_path.resolve())
    print("Evaluation split type:", checkpoint["evaluation_split"])


if __name__ == "__main__":
    main()
