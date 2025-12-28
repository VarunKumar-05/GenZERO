import numpy as np
import librosa
from typing import Optional


def _clip_or_pad(vec: np.ndarray, length: int) -> np.ndarray:
    if vec.shape[0] >= length:
        return vec[:length]
    return np.pad(vec, (0, length - vec.shape[0]), mode="constant")


def extract_audio_signature(
    file_path: str,
    sr: int = 16000,
    mfcc_count: int = 13,
    mel_bins: int = 16,
    chroma_bins: int = 12,
) -> Optional[np.ndarray]:
    """Extracts a compact statistical signature (MFCC, Mel, Chroma, ZCR/RMS)."""
    try:
        y, sr = librosa.load(file_path, sr=sr)
    except Exception as exc:
        print(f"Could not load audio file {file_path}: {exc}")
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_count)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_bins)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=chroma_bins)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    mfcc_mean = _clip_or_pad(mfcc.mean(axis=1), mfcc_count)
    mel_mean = _clip_or_pad(librosa.power_to_db(mel, ref=np.max).mean(axis=1), mel_bins)
    chroma_mean = _clip_or_pad(chroma.mean(axis=1), chroma_bins)
    stats = np.array([
        np.mean(zcr),
        np.std(zcr),
        np.mean(rms),
        np.std(rms),
    ])

    signature = np.concatenate([mfcc_mean, mel_mean, chroma_mean, stats])
    return signature
