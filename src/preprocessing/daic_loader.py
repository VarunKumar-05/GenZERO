import glob
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.input.audio_features import extract_audio_signature

STOP_WORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "but", 
    "of", "to", "in", "for", "with", "by", "as", "it", "this", "that",
    "be", "are", "was", "were", "been", "being", "have", "has", "had",
    "do", "does", "did", "from", "up", "down", "out", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", 
    "where", "why", "how", "all", "any", "both", "each", "few", 
    "more", "most", "other", "some", "such", "no", "nor", "not", 
    "only", "own", "same", "so", "than", "too", "very", "can", 
    "will", "just", "don", "should", "now"
}

POSITIVE_WORDS = {
    "good", "great", "happy", "excellent", "love", "like", "wonderful", 
    "best", "better", "awesome", "joy", "calm", "confident", "fine", 
    "ok", "okay", "glad", "nice", "cool", "yeah", "yes", "sure"
}

NEGATIVE_WORDS = {
    "bad", "sad", "angry", "terrible", "hate", "worst", "worse", 
    "awful", "fear", "anxious", "depressed", "nervous", "upset", 
    "pain", "hurt", "difficult", "hard", "problem", "wrong", 
    "no", "nope", "not", "never"
}

class DAICSession:
    def __init__(self, session_id: str, features: np.ndarray, emphasized_words: List[str], transcript: List[Dict[str, Any]] = None):
        self.session_id = session_id
        self.features = features
        self.emphasized_words = emphasized_words
        self.transcript = transcript or []

class DAICLoader:
    def __init__(
        self,
        root_dir: str,
        target_dim: int = 128,
        audio_enabled: bool = True,
        audio_feature_dim: int = 45,
    ):
        self.root_dir = root_dir
        self.target_dim = target_dim
        self.audio_enabled = audio_enabled
        self.audio_feature_dim = audio_feature_dim
        self.sessions = self._discover_sessions()

    def _discover_sessions(self) -> List[str]:
        paths = []
        for entry in os.listdir(self.root_dir):
            full_path = os.path.join(self.root_dir, entry)
            if os.path.isdir(full_path):
                paths.append(full_path)
        return sorted(paths)

    def _load_transcript(self, transcript_path: str) -> Dict[str, Any]:
        """
        Loads transcript TSV.
        Returns:
            dict containing:
            - 'emphasized_words': list of words (heuristic)
            - 'semantic_features': np.ndarray of shape (feature_dim,)
            - 'emotion_score': float
        """
        results = {
            "emphasized_words": [],
            "semantic_features": np.zeros(5), # count, unique, avg_len, stop_ratio, irrelevant
            "emotion_score": 0.0,
            "transcript_rows": []
        }
        
        if not os.path.exists(transcript_path):
            return results
            
        df = pd.read_csv(transcript_path, sep='\t')
        
        # 1. Participant Filtering: "the sentences with participant should be considered"
        participant_rows = df[df['speaker'].str.lower() == 'participant']
        
        all_words = []
        valid_words = [] # Words that are NOT stop words
        
        sentiment_score = 0
        
        for _, row in participant_rows.iterrows():
            text = str(row['value'])
            start_t = row['start_time']
            stop_t = row['stop_time']
            
            # Store raw row for timeline alignment
            results["transcript_rows"].append({
                "start": start_t,
                "stop": stop_t,
                "text": text
            })

            # clean and tokenize
            tokens = text.lower().split()
            for t in tokens:
                clean_t = t.strip(".,?!")
                if not clean_t:
                    continue
                    
                all_words.append(clean_t)
                
                # 2. Bag of words which do not add weights (Stop Words)
                if clean_t in STOP_WORDS:
                    continue
                
                valid_words.append(clean_t)
                
                # Emotion heuristic
                if clean_t in POSITIVE_WORDS:
                    sentiment_score += 1
                elif clean_t in NEGATIVE_WORDS:
                    sentiment_score -= 1
                    
        # Calculate Semantic Features based on Valid Words
        if valid_words:
            # Simple stats as semantic features
            avg_len = sum(len(w) for w in valid_words) / len(valid_words)
            unique_count = len(set(valid_words))
            total_valid = len(valid_words)
        else:
            avg_len = 0
            unique_count = 0
            total_valid = 0
            
        total_words = len(all_words) if all_words else 1
        stop_word_ratio = (len(all_words) - len(valid_words)) / total_words
        
        # [valid_count, unique_count, avg_len, stop_ratio, sentiment]
        results["semantic_features"] = np.array([
            total_valid, 
            unique_count, 
            avg_len, 
            stop_word_ratio, 
            sentiment_score
        ], dtype=np.float32)
        
        results["emotion_score"] = float(sentiment_score)
        
        # Emphasized words (heuristic for memory module)
        # We only consider valid words (non-stop words) for emphasis tracking too
        results["emphasized_words"] = [w for w in valid_words if len(w) > 3]
        
        return results

    def _load_features(self, covarep_path: str, audio_signature: Optional[np.ndarray], transcript_data: Dict[str, Any]) -> np.ndarray:
        # Acoustic Features (COVAREP)
        # Checks: "check if the words derived are based on frequency (Hertz)"
        # COVAREP features include F0 (Fundamental Frequency) in Hz, VUV (Voiced/Unvoiced), 
        # and Formants (Hz). By loading COVAREP, we verify we are using frequency-based content.
        
        if not os.path.exists(covarep_path):
            acoustic_part = np.zeros((1, 10))
        else:
            df = pd.read_csv(covarep_path, header=None)
            # Downsample
            df_down = df.iloc[::20]
            arr = df_down.to_numpy()
            if arr.size == 0:
                acoustic_part = np.zeros((1, 10))
            else:
                mean = arr.mean(axis=0, keepdims=True)
                std = arr.std(axis=0, keepdims=True) + 1e-6
                acoustic_part = (arr - mean) / std
        
        # Determine dimensions
        audio_dim = audio_signature.shape[0] if audio_signature is not None else 0
        semantic_dim = transcript_data["semantic_features"].shape[0]
        
        # We need to construct the final frame-level feature vectors.
        # Structure: [Acoustic_Frame_t, Audio_Signature_Static, Semantic_Static]
        
        # Calculate how much space we have for acoustic features in the target vector
        # target_dim = acoustic + audio_sig + semantic
        reserved_static = audio_dim + semantic_dim
        covarep_target = max(1, self.target_dim - reserved_static)
        
        # Resize/Pad Acoustic part
        if acoustic_part.shape[1] >= covarep_target:
            covarep_slice = acoustic_part[:, :covarep_target]
        else:
            pad_width = covarep_target - acoustic_part.shape[1]
            covarep_slice = np.pad(acoustic_part, ((0,0),(0,pad_width)), mode='constant')
            
        num_frames = covarep_slice.shape[0]
        
        # Create Static parts (repeated for each frame)
        static_parts = []
        
        # 1. Audio Signature
        if audio_signature is not None and audio_dim > 0:
            static_parts.append(np.tile(audio_signature.reshape(1, -1), (num_frames, 1)))
            
        # 2. Semantic/Emotion Features
        # "include sematic content -feature ... the emotion can be added as one of the factors"
        sem_feat = transcript_data["semantic_features"]
        static_parts.append(np.tile(sem_feat.reshape(1, -1), (num_frames, 1)))
        
        # Combine all
        if static_parts:
            combined_static = np.concatenate(static_parts, axis=1)
            combined = np.concatenate([covarep_slice, combined_static], axis=1)
        else:
            combined = covarep_slice

        # Final safety pad
        if combined.shape[1] < self.target_dim:
            pad_extra = self.target_dim - combined.shape[1]
            combined = np.pad(combined, ((0,0),(0,pad_extra)), mode='constant')
        elif combined.shape[1] > self.target_dim:
             # This might happen if static features are huge, truncate acoustic
             combined = combined[:, :self.target_dim]
             
        return combined

    def _find_audio_file(self, session_path: str) -> Optional[str]:
        for ext in (".wav", ".flac", ".mp3"):
            matches = glob.glob(os.path.join(session_path, f"*{ext}"))
            if matches:
                return matches[0]
        return None

    def _load_audio_signature(self, session_path: str) -> Optional[np.ndarray]:
        if not self.audio_enabled:
            return None
        audio_file = self._find_audio_file(session_path)
        if not audio_file:
            return None
        signature = extract_audio_signature(audio_file)
        if signature is None:
            return None
        if signature.shape[0] > self.audio_feature_dim:
            signature = signature[: self.audio_feature_dim]
        elif signature.shape[0] < self.audio_feature_dim:
            pad_width = self.audio_feature_dim - signature.shape[0]
            signature = np.pad(signature, (0, pad_width), mode='constant')
        return signature

    def iter_sessions(self) -> List[DAICSession]:
        data = []
        for session_path in self.sessions:
            session_id = os.path.basename(session_path)
            transcript_path = os.path.join(session_path, f"{session_id.split('_')[0]}_TRANSCRIPT.csv")
            covarep_path = os.path.join(session_path, f"{session_id.split('_')[0]}_COVAREP.csv")
            
            transcript_results = self._load_transcript(transcript_path)
            audio_signature = self._load_audio_signature(session_path)
            
            features = self._load_features(covarep_path, audio_signature, transcript_results)
            
            data.append(DAICSession(
                session_id=session_id, 
                features=features, 
                emphasized_words=transcript_results["emphasized_words"],
                transcript=transcript_results["transcript_rows"]
            ))
        return data

    def summarize(self) -> Dict[str, Any]:
        return {
            "total_sessions": len(self.sessions),
            "session_ids": [os.path.basename(p) for p in self.sessions]
        }
