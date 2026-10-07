import sys
import streamlit as st
import numpy as np
import scipy.signal as signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# -------------------------------------------------------------------
# 1. CORE DATA LOGIC & ARTIFACT SIMULATION
# -------------------------------------------------------------------
def generate_synthetic_ecg(duration=10.0, fs=250.0, heart_rate=60.0):
    """
    Generates a clean synthetic ECG waveform (P-Q-R-S-T waves).
    Acts as the baseline dataset for testing clinical software logic.
    """
    total_samples = int(duration * fs)
    time = np.linspace(0, duration, total_samples, endpoint=False)
   
    # Basic heart rate tracking frequency
    bps = heart_rate / 60.0
   
    # Simulate basic heartbeat structure using a combined localized sine wave
    # A standard R-peak occurs at every interval cycle
    ecg_clean = np.zeros(total_samples)
    r_peaks = signal.argreaks = np.arange(0, total_samples, int(fs / bps))
   
    for peak in r_peaks:
        # Construct the intense QRS complex spike
        qrs_width = int(0.04 * fs)
        qrs_range = np.arange(max(0, peak-qrs_width), min(total_samples, peak+qrs_width))
        if len(qrs_range) > 0:
            ecg_clean[qrs_range] = np.exp(-((qrs_range - peak) / (0.015 * fs))**2)
           
    return time, ecg_clean

def inject_signal_noise(clean_signal, time, fs, baseline_drift, powerline_noise, white_noise):
    """
    Simulates real-world testing errors by corrupting clinical data
    with baseline wander (low freq) and powerline hum (60Hz).
    """
    # 1. Simulate low-frequency muscle movements / baseline drift (0.5 Hz sine wave)
    drift = baseline_drift * np.sin(2 * np.pi * 0.5 * time)
   
    # 2. Simulate 60Hz powerline hum interference from electrical outlets
    hum = powerline_noise * np.sin(2 * np.pi * 60.0 * time)
   
    # 3. Simulate high-frequency ambient device white noise
    random_noise = white_noise * np.random.normal(0, 1, len(clean_signal))
   
    dirty_signal = clean_signal + drift + hum + random_noise
    return dirty_signal

# -------------------------------------------------------------------
# 2. THE DIGITAL SIGNAL PROCESSING LOOP
# -------------------------------------------------------------------
def apply_bandpass_filter(data, lowcut=0.5, highcut=15.0, fs=250.0, order=2):
    """
    Executes a clinical-grade Scipy Butterworth bandpass filter
    to scrub out baseline drift and high frequency device artifacts.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def pan_tompkins_qrs_detector(filtered_signal, fs):
    """
    Implementation steps of the Pan-Tompkins Algorithm
    to highlight R-peak signal energy for heart rate monitoring.
    """
    # Step A: Derivative calculation to isolate the maximum slope of the R wave
    derivative = np.diff(filtered_signal)
    derivative = np.append(derivative, 0) # Maintain array size consistency
   
    # Step B: Squaring function to exponentially amplify QRS complexes
    squared = derivative ** 2
   
    # Step C: Moving Window Integration to extract broad pulse envelopes
    window_size = int(0.12 * fs)
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
   
    return integrated

# -------------------------------------------------------------------
# 3. USER INTERFACE GRAPHICS ENGINE
# -------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="ECG Signal Live-Debugger")
   
    st.title("🫀 Clinical ECG Signal Live-Debugger & Noise Simulator")
    st.markdown("Use the control sliders in the sidebar to simulate testing errors and monitor filter performance.")
   
    # Dashboard Side Panel Layout
    st.sidebar.header("🛠️ Diagnostic Parameters")
    heart_rate = st.sidebar.slider("Heart Rate (BPM)", 40.0, 140.0, 72.0, step=1.0)
   
    st.sidebar.subheader("⚠️ Injected Test Artifacts")
    baseline_drift = st.sidebar.slider("Baseline Wander Intensity (0.5Hz)", 0.0, 3.0, 1.2, step=0.1)
    powerline_noise = st.sidebar.slider("Powerline Hum Intensity (60Hz)", 0.0, 2.0, 0.5, step=0.1)
    white_noise = st.sidebar.slider("Ambient White Noise (Thermal)", 0.0, 1.0, 0.1, step=0.05)
   
    st.sidebar.subheader("📐 Filter Configuration")
    low_cutoff = st.sidebar.slider("Low Cutoff Frequency (Hz)", 0.1, 2.0, 0.5, step=0.1)
    high_cutoff = st.sidebar.slider("High Cutoff Frequency (Hz)", 10.0, 50.0, 15.0, step=1.0)

    # Core Execution Trigger Loop
    sampling_freq = 250.0
    time, clean_signal = generate_synthetic_ecg(duration=8.0, fs=sampling_freq, heart_rate=heart_rate)
   
    # 1. Corrupt data with slider variables
    corrupted_signal = inject_signal_noise(clean_signal, time, sampling_freq, baseline_drift, powerline_noise, white_noise)
   
    # 2. Run signal cleaning math
    cleaned_signal = apply_bandpass_filter(corrupted_signal, lowcut=low_cutoff, highcut=high_cutoff, fs=sampling_freq)
   
    # 3. Run QRS Peak isolation algorithm
    integrated_energy = pan_tompkins_qrs_detector(cleaned_signal, fs=sampling_freq)
   
    # Plotly Visual Subplot Initialization
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("1. Raw / Corrupted Signal Input",
                                        "2. Post-Filter Cleaned Waveform",
                                        "3. Pan-Tompkins Integrated Energy (Peak Tracker)"))
   
    fig.add_trace(go.Scatter(x=time, y=corrupted_signal, name="Noisy Signal", line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=clean_signal, name="True Baseline", line=dict(color='gray', dash='dash')), row=1, col=1)
   
    fig.add_trace(go.Scatter(x=time, y=cleaned_signal, name="Cleaned Signal", line=dict(color='green')), row=2, col=1)
   
    fig.add_trace(go.Scatter(x=time, y=integrated_energy, name="QRS Energy Pulse", line=dict(color='orange')), row=3, col=1)
   
    fig.update_layout(height=700, showlegend=True, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
