import librosa
import numpy as np

def extract_features(file_name):
    y, sr = librosa.load(file_name, duration=30)
    
    # Chroma feature
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)
    
    # Root Mean Square (RMS)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)
    
    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var = np.var(spectral_bandwidth)
    
    # Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)
    
    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    zero_crossing_rate_var = np.var(zero_crossing_rate)
    
    # Harmony and Perceived Pitch
    harmony, _ = librosa.effects.hpss(y)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = [np.mean(mfcc) for mfcc in mfccs]
    mfccs_var = [np.var(mfcc) for mfcc in mfccs]
    
    features = [
        chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, spectral_centroid_mean,
        spectral_centroid_var, spectral_bandwidth_mean, spectral_bandwidth_var, rolloff_mean,
        rolloff_var, zero_crossing_rate_mean, zero_crossing_rate_var, harmony_mean, harmony_var,
        tempo
    ] + mfccs_mean + mfccs_var
    
    return features