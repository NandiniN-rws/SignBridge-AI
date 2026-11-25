
"""Leakage-aware, CPU-friendly trainer for medical ISL images."""

import argparse
import hashlib
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_groups(samples, root):
    """Keep exact duplicates and images from the same User_ID together."""
    parent = list(range(len(samples)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    signer_seen = {}
    hash_seen = {}

    print("Checking duplicate images and signer IDs...")

    for i, (path, _) in enumerate(samples):
        path_obj = Path(path)
        relative = path_obj.relative_to(root)

        # Find User_2, User2, etc. in the folder path or filename.
        match = re.search(r"user[_ -]?(\d+)", str(relative), re.IGNORECASE)
        if match:
            signer = match.group(1)
            if signer in signer_seen:
                union(i, signer_seen[signer])
            else:
                signer_seen[signer] = i

        # Exact file duplicates must never cross the split.
        digest = hashlib.md5(path_obj.read_bytes()).hexdigest()
        if digest in hash_seen:
            union(i, hash_seen[digest])
        else:
            hash_seen[digest] = i

        if (i + 1) % 2000 == 0:
            print(f"Checked {i + 1}/{len(samples)} images")

    roots = [find(i) for i in range(len(samples))]
    _, groups = np.unique(roots, return_inverse=True)

    print("Unique groups:", len(np.unique(groups)))
    return groups


def make_split(targets, groups):
    """Create an approximately 80/20 split without group leakage."""
    indices = np.arange(len(targets))
    classes = np.unique(targets)
    candidates = []

    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    try:
        for train_idx, val_idx in splitter.split(indices, targets, groups):
            if (len(np.unique(targets[train_idx])) == len(classes)
                    and len(np.unique(targets[val_idx])) == len(classes)):
                difference = abs(len(val_idx) / len(indices) - 0.20)
                candidates.append((difference, train_idx, val_idx))
    except ValueError:
        pass

    if candidates:
        _, train_idx, val_idx = min(candidates, key=lambda x: x[0])
        return train_idx, val_idx

    # Try alternative group splits if stratified grouping cannot cover all classes.
    splitter = GroupShuffleSplit(
        n_splits=100, test_size=0.20, random_state=42
    )

    for train_idx, val_idx in splitter.split(indices, targets, groups):
        if (len(np.unique(targets[train_idx])) == len(classes)
                and len(np.unique(targets[val_idx])) == len(classes)):
            return train_idx, val_idx

    raise RuntimeError(
        "Could not create a leakage-free split containing every class. "
        "Check the User_ID folders and dataset."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument(
        "--output",
        default="deep_learning/checkpoints/signbridge_medical_image_cnn.pt"
    )
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    seed_everything()
    torch.set_num_threads(4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model: frozen ResNet18 + trainable classifier")

    root = Path(args.root).resolve()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset = datasets.ImageFolder(
        str(root), transform=image_transform
    )

    if len(dataset.classes) < 2:
        raise SystemExit("At least two class folders are required.")

    targets = np.array(dataset.targets)
    groups = build_groups(dataset.samples, root)

    train_idx, val_idx = make_split(targets, groups)

    print(f"\nImages: {len(dataset)}")
    print(f"Classes: {len(dataset.classes)}")
    print(f"Training images: {len(train_idx)}")
    print(f"Validation images: {len(val_idx)}")
    print("Classes:", ", ".join(dataset.classes))

    # Verify that no group occurs in both partitions.
    overlap = set(groups[train_idx]) & set(groups[val_idx])
    print("Group leakage:", len(overlap))

    if overlap:
        raise RuntimeError("Group leakage detected. Training stopped.")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print("\nLoading pretrained ResNet18...")
    backbone = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )
    backbone.fc = nn.Identity()
    backbone.eval()

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    backbone = backbone.to(device)

    def extract(loader, description):
        features, labels = [], []
        backbone.eval()

        with torch.inference_mode():
            for batch_no, (images, batch_labels) in enumerate(
                loader, start=1
            ):
                embeddings = backbone(images.to(device))
                features.append(embeddings.cpu())
                labels.append(batch_labels.cpu())

                if batch_no % 50 == 0:
                    print(
                        f"{description}: "
                        f"{batch_no * args.batch_size} images processed"
                    )

        return torch.cat(features), torch.cat(labels)

    print("\nExtracting training features...")
    x_train, y_train = extract(train_loader, "Training")

    print("\nExtracting validation features...")
    x_val, y_val = extract(val_loader, "Validation")

    del backbone

    head = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.35),
        nn.Linear(256, len(dataset.classes))
    ).to(device)

    train_features = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=128,
        shuffle=True
    )

    val_features = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=256,
        shuffle=False
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=0.001, weight_decay=0.001
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    stale = 0

    for epoch in range(args.epochs):
        head.train()

        for xb, yb in train_features:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            optimizer.step()

        head.eval()
        actual, predicted = [], []

        with torch.inference_mode():
            for xb, yb in val_features:
                predictions = head(xb.to(device)).argmax(1)
                predicted.extend(predictions.cpu().tolist())
                actual.extend(yb.tolist())

        val_acc = accuracy_score(actual, predicted)
        val_f1 = f1_score(
            actual, predicted, average="macro", zero_division=0
        )

        scheduler.step(val_f1)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Accuracy: {val_acc:.4f} | Macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            stale = 0

            torch.save({
                "architecture": "resnet18_frozen_features_mlp_head",
                "backbone": "resnet18",
                "image_size": args.image_size,
                "mean": mean,
                "std": std,
                "labels": dataset.classes,
                "head_state_dict": head.state_dict(),
                "best_validation_macro_f1": best_f1,
                "evaluation_split": "signer_and_duplicate_grouped"
            }, output)

        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping.")
                break

    checkpoint = torch.load(
        output, map_location=device, weights_only=False
    )
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    actual, predicted = [], []

    with torch.inference_mode():
        for xb, yb in val_features:
            predicted.extend(
                head(xb.to(device)).argmax(1).cpu().tolist()
            )
            actual.extend(yb.tolist())

    print("\n===== GROUPED VALIDATION RESULTS =====")
    print("Accuracy:", accuracy_score(actual, predicted))
    print(
        "Macro F1:",
        f1_score(actual, predicted, average="macro", zero_division=0)
    )
    print(classification_report(
        actual,
        predicted,
        labels=np.arange(len(dataset.classes)),
        target_names=dataset.classes,
        zero_division=0
    ))
    print("Checkpoint saved:", output.resolve())
    print("Evaluation: signer-and-duplicate-grouped validation")


if __name__ == "__main__":
    main()
