
# SignBridge AI – updated dataset/training guide

## What this update adds
- Video recognition architecture: temporal **CNN + BiLSTM + attention** (32 frames × 258 MediaPipe features).
- Separate static-image CNN training for the medical-word image dataset.
- Manifest builder for class-folder datasets.
- Learn ISL dictionary page with the vocabulary labels described by the two datasets.
- Doctor response language selector: English, Kannada and Hindi. Doctor speech recognition and browser speech output use the selected language; GenAI translation requires Ollama and must be reviewed.

## Important: keep datasets on your computer
Do NOT upload large datasets to ChatGPT or put raw datasets inside a distributable ZIP. Keep them locally and do not commit them to GitHub unless the licence permits redistribution.

Create:
```
SignBridge_AI/
  datasets/
    emergency_isl_videos/    # extract emergency video dataset here
      accident/
      call/
      doctor/
      help/
      hot/
      lose/
      pain/
      thief/
    medical_isl_images/      # extract medical image dataset here
      afraid/
      agree/
      assistance/
      ...
```
Each sign must have its own folder. If the downloaded archives have a different structure, do not rename blindly; inspect the folders and build the CSV manifest using `deep_learning/data/manifest_template.csv`.

## Install training dependencies
Open a terminal in `SignBridge_AI`:
```
backend\.venv\Scripts\activate
pip install -r deep_learning\requirements-vision.txt
pip install -r deep_learning\requirements-training.txt
```
For image CNN training, torchvision must match your installed PyTorch version. If pip reports a Torch/torchvision mismatch, install the matching pair from the official PyTorch selector.

## 1) Build video manifest
```
python deep_learning\preprocessing\build_manifests.py --videos-root datasets\emergency_isl_videos
```
This creates `deep_learning/data/manifest_emergency.csv`.
Check that each row has the correct label and participant ID. If participant IDs are not present in the directory structure, the script writes `unknown`; do not claim signer-independent testing.

## 2) Extract video landmarks
```
python deep_learning\preprocessing\extract_landmarks.py --manifest deep_learning\data\manifest_emergency.csv --output deep_learning\data\landmarks_emergency.npz
```
This uses MediaPipe Holistic and stores 32-frame sequences with 258 features per frame.

## 3) Train video CNN-LSTM
```
python deep_learning\training\train.py --data deep_learning\data\landmarks_emergency.npz --epochs 40
```
Checkpoint: `deep_learning/checkpoints/signbridge_video_cnn_lstm.pt`.
For backend inference, set `SIGN_MODEL_PATH` in `.env` to this exact checkpoint path. The backend config default may point elsewhere; confirm before launching.

## 4) Train static medical image CNN
```
python deep_learning\training\train_medical_images.py --root datasets\medical_isl_images --epochs 15
```
Checkpoint: `deep_learning/checkpoints/signbridge_medical_image_cnn.pt`.
This is a separate static-image model. It is not a temporal LSTM model and is not yet wired into live webcam inference.

## 5) Run app
From project root:
```
start_signbridge.bat
```
Open `http://localhost:5173`. Backend docs: `http://127.0.0.1:8000/docs`.

## GenAI and language setup
Install Ollama separately, pull a supported model, and ensure it is running. Set `OLLAMA_MODEL` and `OLLAMA_BASE_URL` in `.env`. If Ollama is unavailable, the app retains original text; it must not label fallback text as translated. Confirm all generated/translated clinical messages before sending.

## Dataset and evaluation cautions
- The two datasets have different modalities: emergency dataset is videos; medical dataset is images. They cannot be merged into one temporal training tensor without fabricating motion.
- Confirm the dataset's actual licence and attribution requirements from each original download page before redistributing any samples.
- Use participant-wise splits whenever participant IDs are supplied. If IDs are absent, report the evaluation as a sample-level split and a limitation, not signer-independent performance.
- The Learn ISL page explains gloss meanings; it does not yet show verified sign demonstration media. Use original dataset examples or a vetted ISL dictionary for demonstrations.
- These datasets cover a limited vocabulary and are not continuous sentence-level ISL translation. The prototype must request patient confirmation and use a qualified interpreter for clinical communication.
