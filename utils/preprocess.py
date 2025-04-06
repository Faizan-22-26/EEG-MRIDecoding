import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

class EEGNormalizer:
    """
    A robust and modular EEG normalizer that supports multiple strategies.
    """

    def __init__(self, method="zscore"):
        """
        method: str
            Supported: 'zscore', 'robust', 'minmax'
        """
        self.method = method
        self.scaler = None

    def fit(self, signal: np.ndarray):
        """
        Fit the normalization strategy to the signal.
        """
        if self.method == "zscore":
            self.mean = np.mean(signal)
            self.std = np.std(signal)
        elif self.method == "robust":
            self.scaler = RobustScaler().fit(signal.reshape(-1, 1))
        elif self.method == "minmax":
            self.min = np.min(signal)
            self.max = np.max(signal)

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Transform signal using fitted strategy.
        """
        if self.method == "zscore":
            return (signal - self.mean) / (self.std + 1e-8)
        elif self.method == "robust":
            return self.scaler.transform(signal.reshape(-1, 1)).flatten()
        elif self.method == "minmax":
            return (signal - self.min) / (self.max - self.min + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        self.fit(signal)
        return self.transform(signal)
