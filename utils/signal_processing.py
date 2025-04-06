import numpy as np
import scipy.signal as sp

class EEGFilterPipeline:
    """
    A customizable EEG signal processing pipeline.
    """

    def __init__(self, fs=256):
        self.fs = fs
        self.pipeline = []

    def add_bandpass(self, low=1, high=50, order=5):
        def bandpass(signal):
            b, a = sp.butter(order, [low / (self.fs/2), high / (self.fs/2)], btype='band')
            return sp.filtfilt(b, a, signal)
        self.pipeline.append(('bandpass', bandpass))
        return self

    def add_notch(self, freq=50, q=30):
        def notch(signal):
            b, a = sp.iirnotch(freq, q, self.fs)
            return sp.filtfilt(b, a, signal)
        self.pipeline.append(('notch', notch))
        return self

    def add_detrend(self):
        def detrend(signal):
            return sp.detrend(signal)
        self.pipeline.append(('detrend', detrend))
        return self

    def run(self, signal: np.ndarray) -> np.ndarray:
        for name, step in self.pipeline:
            signal = step(signal)
        return signal
