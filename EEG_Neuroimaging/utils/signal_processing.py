import scipy.signal as signal

def bandpass_filter(eeg, low=1.0, high=50.0, fs=256):
    b, a = signal.butter(4, [low/fs*2, high/fs*2], btype='band')
    return signal.filtfilt(b, a, eeg)
