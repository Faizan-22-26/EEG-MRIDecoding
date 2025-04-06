from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model("model/saved_model/")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    signal = np.frombuffer(data, dtype=np.float32).reshape(1, -1, 1)
    prediction = model.predict(signal).tolist()
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
