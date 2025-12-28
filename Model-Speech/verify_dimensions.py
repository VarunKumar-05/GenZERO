import numpy as np
import os
import sys

# Mocking things to test logic without files
from src.preprocessing.daic_loader import DAICLoader, STOP_WORDS, POSITIVE_WORDS, NEGATIVE_WORDS

def test_logic():
    print("Testing DAICLoader logic...")
    if not os.path.exists("dummy_root"):
        os.makedirs("dummy_root")
    loader = DAICLoader("dummy_root", target_dim=128, audio_enabled=True)
    
    # Test _load_transcript with dummy file (mocked by overwriting the method or creating a temp file)
    # We will just test the logic inside if we could, but since it reads a file, let's create a temporary dummy file.
    
    import pandas as pd
    
    # Create dummy transcript
    transcript_content = {
        "speaker": ["participant", "interviewer", "participant"],
        "value": ["Hello world this is a test.", "Okay.", "I feel good and happy but also a bit sad."]
    }
    df = pd.DataFrame(transcript_content)
    df.to_csv("test_transcript.csv", sep='\t', index=False)
    
    # Test _load_transcript
    print("Testing _load_transcript...")
    results = loader._load_transcript("test_transcript.csv")
    print("Results:", results)
    
    # Verify Stop Words
    # "this", "is", "a" are in STOP_WORDS. "hello", "world", "test" are not.
    # "i", "feel", "good", "and", "happy", "but", "also", "a", "bit", "sad"
    # "i" not in defaults? Let's check. 
    # STOP_WORDS in code: "the", "is", "at", "which", "on", "a", "an", "and", "or", "but", ...
    
    sem_feat = results["semantic_features"]
    print("Semantic Features:", sem_feat)
    # Check Sentiment
    # "good", "happy" -> +2. "sad" -> -1. Total = 1.
    if results["emotion_score"] == 1.0:
        print("PASS: Emotion score correct.")
    else:
        print(f"FAIL: Emotion score {results['emotion_score']} != 1.0")

    # Test _load_features combination
    print("\nTesting _load_features dimensions...")
    # Mock inputs
    # Acoustic: 100 frames, 10 dims
    # We can't easily mock the file reading inside _load_features without creating a file.
    # Let's create a dummy covarep file.
    covarep_data = np.random.randn(100, 74) # 74 is visible in original code? No, usually 74.
    np.savetxt("test_covarep.csv", covarep_data, delimiter=",")
    
    # Audio sig
    audio_sig = np.random.randn(45)
    
    # Run
    features = loader._load_features("test_covarep.csv", audio_sig, results)
    print("Output features shape:", features.shape)
    
    if features.shape[1] == 128:
        print("PASS: Output dimension is 128.")
    else:
        print(f"FAIL: Output dimension {features.shape[1]} != 128")
        
    # Clean up
    if os.path.exists("test_transcript.csv"): os.remove("test_transcript.csv")
    if os.path.exists("test_covarep.csv"): os.remove("test_covarep.csv")

if __name__ == "__main__":
    test_logic()
