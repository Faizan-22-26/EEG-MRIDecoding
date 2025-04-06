import numpy as np

def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)
