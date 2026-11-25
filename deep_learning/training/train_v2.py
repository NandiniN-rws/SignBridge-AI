
import argparse
import csv
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


# ---------------- CONFIGURATION ----------------

SEED = 42
NUM_FRAMES = 16
IMAGE_SIZE = 64
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.0003

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "deep_learning" / "data" / "manifest_emergency.csv"
CHECKPOINT = ROOT / "deep_learning" / "checkpoints" / "signbridge_emergency_cnn_lstm_transformer.pt"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- VIDEO PREPROCESSING ----------------

def load_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 1:
        cap.release()
        raise ValueError(f"Video has no readable frames: {video_path}")

    positions = np.linspace(
        0, total_frames - 1, NUM_FRAMES
    ).astype(int)

    frames = []

    for position in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        success, frame = cap.read()

        if not success:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(
                    np.zeros(
                        (IMAGE_SIZE, IMAGE_SIZE, 3),
                        dtype=np.uint8
                    )
                )
            continue

        height, width = frame.shape[:2]

        # Resize while preserving the original aspect ratio.
        scale = IMAGE_SIZE / max(height, width)

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        canvas = np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE, 3),
            dtype=np.uint8
        )

        x_offset = (IMAGE_SIZE - new_width) // 2
        y_offset = (IMAGE_SIZE - new_height) // 2

        canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width
        ] = frame

        frame = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2RGB
        )

        frames.append(frame)

    cap.release()

    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1].copy())

    return np.stack(frames[:NUM_FRAMES])


# ---------------- DATASET ----------------

class EmergencyVideoDataset(Dataset):

    def __init__(self, videos, labels):
        self.videos = videos
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        frames = self.videos[index].astype(np.float32) / 255.0

        # T,H,W,C -> T,C,H,W
        frames = torch.from_numpy(
            frames.transpose(0, 3, 1, 2).copy()
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return frames, label


# ---------------- CNN + BILSTM + TRANSFORMER ----------------

class FrameCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(128, 128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)


class CNNLSTMTransformer(nn.Module):

    def __init__(self, num_classes, num_frames=NUM_FRAMES):
        super().__init__()

        self.cnn = FrameCNN()

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_frames, 256)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            dropout=0.2,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        batch, frames, channels, height, width = x.shape

        x = x.reshape(
            batch * frames,
            channels,
            height,
            width
        )

        x = self.cnn(x)

        x = x.reshape(
            batch,
            frames,
            128
        )

        x, _ = self.lstm(x)

        x = x + self.position_embedding[:, :frames, :]

        x = self.transformer(x)

        # Temporal average pooling.
        x = x.mean(dim=1)

        return self.classifier(x)


# ---------------- MANIFEST AND SPLIT ----------------

def read_manifest(manifest_path):

    with open(
        manifest_path,
        encoding="utf-8-sig",
        newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    videos = []
    labels = []
    groups = []

    for row in rows:

        video_path = Path(row["video_path"])

        if not video_path.is_absolute():
            video_path = manifest_path.parent / video_path

        if not video_path.exists():
            print("Skipping missing video:", video_path)
            continue

        label = row["label"].strip()

        if not label:
            continue

        videos.append(video_path)
        labels.append(label)

        signer = row.get("signer_id", "").strip()

        if not signer or signer.lower() in ("unknown", "none", "nan"):
            match = re.search(
                r"user[_ -]?(\d+)",
                str(video_path),
                re.IGNORECASE
            )

            signer = (
                f"user_{match.group(1)}"
                if match else ""
            )

        groups.append(signer)

    return videos, labels, groups


def make_split(labels, groups):

    labels = np.asarray(labels)
    groups = np.asarray(groups)

    indices = np.arange(len(labels))

    # Use signer-based grouping when IDs are available.
    if len(set(groups)) > 1 and np.all(groups != ""):

        splitter = GroupShuffleSplit(
            n_splits=30,
            test_size=0.20,
            random_state=SEED
        )

        all_classes = set(np.unique(labels))

        for train_idx, val_idx in splitter.split(
            indices,
            labels,
            groups
        ):

            if (
                set(np.unique(labels[train_idx])) == all_classes
                and set(np.unique(labels[val_idx])) == all_classes
            ):
                print("Split method: signer-grouped validation")
                return train_idx, val_idx

        print(
            "WARNING: Could not create a signer split "
            "containing every class in both sets."
        )

    # Fallback: stratified video-level split.
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED
    )

    train_idx, val_idx = next(
        splitter.split(indices, labels)
    )

    print("Split method: stratified video-level validation")
    print("WARNING: Independent signer validation is not guaranteed.")

    return train_idx, val_idx


# ---------------- TRAINING ----------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST)
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE
    )

    args = parser.parse_args()

    set_seed()

    torch.set_num_threads(2)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    manifest_path = Path(args.manifest).resolve()

    print("\n========== SIGNBRIDGE EMERGENCY V2 ==========")
    print("Architecture: CNN + BiLSTM + Transformer")
    print("Device:", device)

    video_paths, raw_labels, groups = read_manifest(
        manifest_path
    )

    if len(video_paths) < 10:
        raise RuntimeError("Not enough valid videos to train.")

    classes = sorted(set(raw_labels))

    label_to_id = {
        label: index
        for index, label in enumerate(classes)
    }

    y = np.asarray([
        label_to_id[label]
        for label in raw_labels
    ])

    print("Videos:", len(video_paths))
    print("Classes:", classes)

    print("\nLoading video frames...")

    X = []

    for index, path in enumerate(video_paths, start=1):

        try:
            frames = load_video(path)
            X.append(frames)

        except Exception as error:
            print("Skipping:", path, error)
            X.append(
                np.zeros(
                    (
                        NUM_FRAMES,
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                        3
                    ),
                    dtype=np.uint8
                )
            )

        if index % 25 == 0:
            print(f"Loaded {index}/{len(video_paths)} videos")

    X = np.stack(X)

    print("Video tensor:", X.shape)

    train_idx, val_idx = make_split(
        raw_labels,
        groups
    )

    X_train = X[train_idx]
    X_val = X[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    print("Training videos:", len(train_idx))
    print("Validation videos:", len(val_idx))

    train_dataset = EmergencyVideoDataset(
        X_train,
        y_train
    )

    val_dataset = EmergencyVideoDataset(
        X_val,
        y_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = CNNLSTMTransformer(
        num_classes=len(classes)
    ).to(device)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = np.ones(
        len(classes),
        dtype=np.float32
    )

    for class_id, weight in zip(
        np.unique(y_train),
        weights
    ):
        class_weights[class_id] = weight

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32
        ).to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_accuracy = 0.0
    patience = 5
    no_improvement = 0

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    for epoch in range(args.epochs):

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for frames, labels in train_loader:

            frames = frames.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(frames)

            loss = criterion(outputs, labels)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        train_accuracy = (
            train_correct / max(train_total, 1)
        )

        model.eval()

        val_correct = 0
        val_total = 0
        val_loss = 0.0

        all_predictions = []
        all_targets = []

        with torch.no_grad():

            for frames, labels in val_loader:

                frames = frames.to(device)
                labels = labels.to(device)

                outputs = model(frames)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

                all_predictions.extend(
                    predictions.cpu().numpy().tolist()
                )

                all_targets.extend(
                    labels.cpu().numpy().tolist()
                )

        val_accuracy = (
            val_correct / max(val_total, 1)
        )

        scheduler.step(val_accuracy)

        print(
            f"\nEpoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss / max(len(train_loader), 1):.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss / max(len(val_loader), 1):.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy
            no_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "num_frames": NUM_FRAMES,
                    "image_size": IMAGE_SIZE,
                    "architecture": "CNN_BiLSTM_Transformer",
                    "validation_accuracy": best_accuracy,
                    "label_to_id": label_to_id
                },
                CHECKPOINT
            )

            print("Best checkpoint saved.")

        else:

            no_improvement += 1

        if no_improvement >= patience:

            print("\nEarly stopping.")
            break

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print("Checkpoint:", CHECKPOINT.resolve())

    # Save final validation report.
    from sklearn.metrics import classification_report

    report = classification_report(
        all_targets,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
        output_dict=True
    )

    report_path = CHECKPOINT.parent / "emergency_v2_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Validation report:", report_path.resolve())


if __name__ == "__main__":
    main()
