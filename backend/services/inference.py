import numpy as np
import tensorflow as tf
from backend.utils.signal_pipeline import get_preprocessing_pipeline

class InferenceService:
    def __init__(self):
        self.model = tf.keras.models.load_model("model/saved_model/")
        self.pipeline = get_preprocessing_pipeline()

    def run_inference(self, raw_bytes: bytes) -> list:
        signal = np.frombuffer(raw_bytes, dtype=np.float32)

        processed = self.pipeline.process(signal)
        model_input = processed.reshape(1, -1, 1)

        output = self.model.predict(model_input)
        return output.flatten().tolist()
