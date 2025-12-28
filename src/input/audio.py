import librosa
import numpy as np

class AudioStream:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        print(f"Initialized Audio Stream with SR={sample_rate}")

    def load_file(self, file_path):
        """Loads an audio file."""
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            return y, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None

    def extract_features(self, audio_data):
        """Extracts features like MFCC, Zero Crossing Rate, and RMS."""
        if audio_data is None:
            return None
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        
        # Zero Crossing Rate (proxy for high frequency noise/tremor in voice)
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        
        # RMS Energy (loudness)
        rms = librosa.feature.rms(y=audio_data)
        
        # Return as a dictionary of time-series features
        features = {
            "mfcc": mfccs.T, # Transpose to (Time, Features)
            "zcr": zcr.T,
            "rms": rms.T
        }
        return features

    def get_data(self, file_path=None):
        if file_path:
            y, _ = self.load_file(file_path)
            return self.extract_features(y)
        else:
            # Simulate audio data for testing
            print("Warning: No file path provided, returning simulated data.")
            return {"type": "audio", "data": [0.1, 0.2, -0.1]}
