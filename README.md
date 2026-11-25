# SignBridge AI
Local development application for accessible hospital communication.

## Quick start (Windows)
1. Install Python 3.11 and Node.js 20+.
2. Extract this ZIP and open `SignBridge_AI` in VS Code.
3. Run `setup.bat`.
4. Run `start_signbridge.bat`.
5. Open http://localhost:5173 and API docs at http://127.0.0.1:8000/docs.

Optional GenAI: install Ollama from https://ollama.com, run `ollama pull llama3.2:3b`, then `ollama serve`.

## ISL recognition: important limitation
No pretrained checkpoint or third-party dataset is included. The app will explicitly show MODEL_NOT_READY until you train a checkpoint. It never fabricates sign predictions. The PyTorch BiLSTM-attention architecture, MediaPipe landmark extraction, manifest format, and training/evaluation script are included.

For dataset training, obtain INCLUDE or another ISL dataset from its legitimate source and check its licence. Do not redistribute dataset videos without permission. Use `deep_learning/data/manifest_template.csv`, extract landmarks, then run:
```
pip install -r deep_learning/requirements-training.txt
pip install -r deep_learning/requirements-vision.txt
python deep_learning/preprocessing/extract_landmarks.py --manifest deep_learning/data/manifest.csv --output deep_learning/data/landmarks.npz
python deep_learning/training/train.py --data deep_learning/data/landmarks.npz --epochs 40
```
The manifest columns are `video_path,label,signer_id,split`. Use anonymized signer IDs. The extractor samples 32 frames; each frame has 258 values (pose 33x4 + two hands 21x3 each). Training attempts signer-grouped splits when at least three signer groups are available. Small datasets can overfit; report actual held-out metrics only.

## Included and external requirements
- React/Vite interface, FastAPI backend, SQLite conversation history.
- Browser microphone speech recognition (browser support dependent) and browser text-to-speech.
- Optional Ollama message improvement/simplification with a clearly labelled no-LLM fallback.
- MediaPipe extraction and PyTorch train/evaluate pipeline.
- Camera preview works in the browser. A trained checkpoint is needed for recognition; connect a landmark producer to `/api/sign/predict` using exactly 32x258 float features.
- This is not a certified medical interpreter, diagnosis system, or emergency response replacement. Human confirmation is required. The keyword emergency flag is not clinical triage.
- Production authentication, encryption, retention, clinical validation, and regulatory review are not implemented. Do not use real patient data in this local demo.

Backend tests: `cd backend && pip install -r requirements.txt && pytest`
Frontend build: `cd frontend && npm install && npm run build`


## Dataset training update
See [DATASET_TRAINING_GUIDE.md](DATASET_TRAINING_GUIDE.md) for local dataset folder setup, landmark extraction, video CNN-LSTM training, static image CNN training, Learn ISL and multilingual doctor responses.
