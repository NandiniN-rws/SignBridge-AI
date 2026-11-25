
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # Groq cloud AI configuration
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-20b"

    # Sign recognition models
    sign_model_path: str = str(ROOT / "deep_learning/checkpoints/signbridge_emergency_cnn_lstm_transformer.pt")

    medical_image_model_path: str = str(
        ROOT / "deep_learning/checkpoints/signbridge_medical_image_cnn.pt"
    )

    confidence_threshold: float = 0.65
    medical_confidence_threshold: float = 0.65

    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173"
    )

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        extra="ignore"
    )


settings = Settings()
