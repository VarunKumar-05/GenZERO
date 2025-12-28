import re

class TextInput:
    def __init__(self):
        print("Initialized Text Input")

    def parse_chat_transcript(self, file_path):
        """Parses a CHAT format transcript file."""
        transcript = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('*'):
                        # Extract speaker and text
                        match = re.match(r'\*(\w+):\s+(.*)', line)
                        if match:
                            speaker = match.group(1)
                            text = match.group(2)
                            transcript.append({"speaker": speaker, "text": text})
            return transcript
        except Exception as e:
            print(f"Error reading transcript: {e}")
            return []

    def vectorize_text(self, text_data):
        """
        Placeholder for text vectorization.
        In a real BDH implementation, this would convert text to non-negative vectors.
        """
        # Simple dummy vectorization (e.g., length of words)
        vectors = []
        for entry in text_data:
            words = entry['text'].split()
            # Dummy vector: [word_count, char_count, 0, 0, ...]
            vec = [len(words), len(entry['text']), 0.0, 0.0] 
            vectors.append(vec)
        return vectors

    def get_data(self, file_path=None):
        if file_path:
            transcript = self.parse_chat_transcript(file_path)
            vectors = self.vectorize_text(transcript)
            return {"transcript": transcript, "vectors": vectors}
        else:
            # Simulate text input
            return {"type": "text", "content": "Patient reports mild tremor."}
