from fastapi import FastAPI
from backend.routers import prediction
from backend.utils.logger import setup_logger
from backend.utils.exceptions import add_custom_exception_handlers

app = FastAPI(
    title="NeuroAPI",
    description="EEG-based real-time neuroimaging inference system",
    version="1.0.0"
)

# Include routers
app.include_router(prediction.router)

# Setup logging
setup_logger()

# Custom exceptions
add_custom_exception_handlers(app)

@app.get("/")
async def root():
    return {"message": "Welcome to the EEG Neuroimaging API"}
