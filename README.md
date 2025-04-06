# EEG-Based Neuroimaging System

A low-cost, portable, and real-time alternative to MRI using EEG signals and AI.

## Features
- Real-time brainwave analysis from EEG data
- ML-powered signal translation into interpretable outputs
- Web and mobile-ready interface
- FastAPI backend + TensorFlow model + HTML/CSS/JS frontend

## Folder Structure
- `model/`: ML model definition and training
- `backend/`: FastAPI server for inference
- `frontend/`: UI to upload EEG files and view predictions
- `utils/`: Signal filtering and preprocessing
- `data/`: Placeholder for EEG recordings (ignored in Git)

## To Run
```bash
uvicorn backend.main:app --reload

