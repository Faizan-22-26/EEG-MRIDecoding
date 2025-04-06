from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.services.inference import InferenceService
from backend.utils.file_validator import validate_eeg_file

router = APIRouter(prefix="/predict", tags=["Prediction"])

service = InferenceService()

@router.post("/")
async def predict_eeg(file: UploadFile = File(...)):
    if not validate_eeg_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file format. Use .bin or .npy")

    raw_data = await file.read()
    prediction = service.run_inference(raw_data)

    return {
        "prediction": prediction,
        "status": "success"
    }
