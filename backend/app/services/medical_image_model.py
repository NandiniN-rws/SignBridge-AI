import base64
import binascii

from io import BytesIO
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

from ..core.config import settings


class MedicalImagePredictor:

    def __init__(self):

        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None
        self.labels = []
        self.error = None

        self.image_size = 160

        self.mean = [
            0.485,
            0.456,
            0.406
        ]

        self.std = [
            0.229,
            0.224,
            0.225
        ]

        path = Path(settings.medical_image_model_path)

        if not path.exists():

            self.error = f"Checkpoint not found: {path}"
            return

        try:

            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False
            )

            if checkpoint.get("architecture") != "resnet18_frozen_features_mlp_head":

                raise ValueError(
                    "Unexpected checkpoint architecture."
                )

            self.labels = list(checkpoint["labels"])

            self.image_size = int(
                checkpoint.get("image_size", 160)
            )

            self.mean = checkpoint.get(
                "mean",
                self.mean
            )

            self.std = checkpoint.get(
                "std",
                self.std
            )

            backbone = models.resnet18(weights=None)

            backbone.fc = nn.Identity()

            head = nn.Sequential(

                nn.Linear(512, 256),

                nn.ReLU(),

                nn.Dropout(0.35),

                nn.Linear(
                    256,
                    len(self.labels)
                )
            )

            # Load the pretrained ResNet18 weights
            # from the local PyTorch cache.

            cache = (
                Path(torch.hub.get_dir())
                / "checkpoints"
                / "resnet18-f37072fd.pth"
            )

            if not cache.exists():

                raise FileNotFoundError(
                    "Cached ResNet18 pretrained weights are missing. "
                    "Run the medical image training script on this computer first."
                )

            backbone_weights = torch.load(
                cache,
                map_location="cpu",
                weights_only=True
            )

            backbone_weights = {
                key: value
                for key, value in backbone_weights.items()
                if not key.startswith("fc.")
            }

            backbone.load_state_dict(
                backbone_weights,
                strict=False
            )

            self.model = nn.Sequential(
                backbone,
                head
            ).to(self.device)

            self.model[1].load_state_dict(
                checkpoint["head_state_dict"]
            )

            self.model.eval()

        except Exception as exc:

            self.model = None
            self.labels = []
            self.error = str(exc)

    def predict_base64(self, image_base64: str):

        if self.model is None:

            return {
                "status": "MODEL_NOT_READY",
                "prediction": None,
                "message": self.error
            }

        try:

            encoded = image_base64.split(",", 1)[-1]

            raw = base64.b64decode(
                encoded,
                validate=True
            )

            if len(raw) > 10 * 1024 * 1024:

                return {
                    "status": "INVALID_IMAGE",
                    "prediction": None,
                    "message": "Image must be 10 MB or smaller."
                }

            image = Image.open(
                BytesIO(raw)
            ).convert("RGB")

        except (
            binascii.Error,
            ValueError,
            OSError
        ):

            return {
                "status": "INVALID_IMAGE",
                "prediction": None,
                "message": "Could not decode the supplied image."
            }

        transform = transforms.Compose([

            transforms.Resize(
                (self.image_size, self.image_size)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                self.mean,
                self.std
            )

        ])

        tensor = transform(
            image
        ).unsqueeze(0).to(self.device)

        with torch.inference_mode():

            probabilities = torch.softmax(
                self.model(tensor),
                dim=1
            )[0]

            confidence, index = probabilities.max(0)

        score = float(confidence.item())

        label = self.labels[
            int(index.item())
        ]

        if score < settings.medical_confidence_threshold:

            return {
                "status": "UNCERTAIN",
                "prediction": None,
                "candidate": label,
                "confidence": score,
                "message": (
                    "Low confidence. Ask the patient "
                    "to repeat or confirm the sign."
                )
            }

        return {
            "status": "PREDICTED_UNCONFIRMED",
            "prediction": label,
            "confidence": score,
            "message": (
                "Image classification only. "
                "Confirm the meaning with the patient."
            )
        }