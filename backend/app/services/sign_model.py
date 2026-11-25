
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..core.config import settings


# Must match train_v2.py exactly.
NUM_FRAMES = 16
IMAGE_SIZE = 64


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

        x = x.reshape(batch, frames, 128)

        x, _ = self.lstm(x)

        x = x + self.position_embedding[:, :frames, :]

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.classifier(x)


class Predictor:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None
        self.labels = []
        self.model_ready = False

        checkpoint_path = Path(settings.sign_model_path)

        if not checkpoint_path.exists():
            print("V2 checkpoint not found:", checkpoint_path)
            return

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False
            )

            self.labels = checkpoint["classes"]

            self.model = CNNLSTMTransformer(
                num_classes=len(self.labels),
                num_frames=checkpoint.get(
                    "num_frames", NUM_FRAMES
                )
            ).to(self.device)

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            self.model.eval()
            self.model_ready = True

            print("SignBridge V2 loaded successfully.")
            print("Architecture: CNN + BiLSTM + Transformer")
            print("Classes:", self.labels)
            print("Device:", self.device)

        except Exception as error:
            self.model = None
            self.labels = []
            self.model_ready = False

            print("V2 model loading failed:", error)

    def predict(self, frames):

        if not self.model_ready:
            return {
                "status": "MODEL_NOT_READY",
                "prediction": None,
                "message": "Emergency V2 model could not be loaded."
            }

        try:
            frames = np.asarray(frames, dtype=np.float32)

            expected_shape = (
                NUM_FRAMES,
                IMAGE_SIZE,
                IMAGE_SIZE,
                3
            )

            if frames.shape != expected_shape:
                return {
                    "status": "INVALID_INPUT",
                    "prediction": None,
                    "expected_shape": expected_shape,
                    "received_shape": frames.shape,
                    "message": "Expected 16 RGB frames of size 64x64."
                }

            # Same normalization and channel ordering as training.
            if frames.max() > 1.0:
                frames = frames / 255.0

            frames = np.clip(frames, 0.0, 1.0)

            # T,H,W,C -> 1,T,C,H,W
            tensor = torch.from_numpy(
                frames.transpose(0, 3, 1, 2).copy()
            ).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                logits = self.model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0]

            score, index = probabilities.max(dim=0)

            confidence = float(score.item())
            prediction = self.labels[int(index.item())]

            threshold = settings.confidence_threshold

            if confidence < threshold:
                return {
                    "status": "UNCERTAIN",
                    "prediction": None,
                    "confidence": confidence,
                    "message": "Low confidence. Please repeat the sign."
                }

            return {
                "status": "PREDICTED_UNCONFIRMED",
                "prediction": prediction,
                "confidence": confidence,
                "message": (
                    "Emergency sign detected. "
                    "Please confirm the message with the patient."
                )
            }

        except Exception as error:
            return {
                "status": "PREDICTION_ERROR",
                "prediction": None,
                "message": str(error)
            }
