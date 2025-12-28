class PersistentMemory:
    def __init__(self):
        print("Initialized Persistent Memory Storage")
        self.storage = {}
        self.word_counts = {}

    def save_state(self, state):
        print("Saving synaptic state...")
        self.storage["last_state"] = state

    def get_baseline(self):
        return {"baseline_metric": 0.5}

    def track_words(self, words):
        """Stores emphasized words with running counts."""
        for w in words:
            key = w.lower().strip()
            if not key:
                continue
            self.word_counts[key] = self.word_counts.get(key, 0) + 1

    def get_top_words(self, k=10):
        if not self.word_counts:
            return []
        # Sort by frequency descending
        return sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)[:k]
