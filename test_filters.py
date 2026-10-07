import numpy as np
from app import apply_bandpass_filter

def test_bandpass_filter_attenuation():
    """
    Verifies that high-frequency noise is successfully attenuated
    by the custom signal processing library.
    """
    fs = 250.0
    t = np.linspace(0, 1, int(fs), endpoint=False)
   
    # Create a clean low frequency baseline signal (1 Hz) + a heavy high-frequency noise spike (80 Hz)
    low_freq = np.sin(2 * np.pi * 1.0 * t)
    high_freq_noise = 2.0 * np.sin(2 * np.pi * 80.0 * t)
    input_signal = low_freq + high_freq_noise
   
    # Process signal through app filter parameters (Cutoff at 15Hz)
    output_signal = apply_bandpass_filter(input_signal, lowcut=0.5, highcut=15.0, fs=fs)
   
    # Check that high frequency distortion energy was cut down considerably
    assert np.max(output_signal) < np.max(input_signal)
    assert len(output_signal) == len(input_signal)
