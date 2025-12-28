import torch
import numpy as np

class SpikeEncoder:
    """
    Converts continuous time-series data into discrete spike trains.
    """
    def __init__(self, method='rate', gain=1.0, offset=0.0):
        self.method = method
        self.gain = gain
        self.offset = offset

    def __call__(self, x):
        if self.method == 'rate':
            return self._rate_coding(x)
        elif self.method == 'latency':
            return self._latency_coding(x)
        else:
            raise ValueError(f"Unknown encoding method: {self.method}")

    def _rate_coding(self, x):
        """
        Rate coding: Probability of firing is proportional to intensity.
        Input x: (Time, Channels)
        Output: (Time, Channels) - Binary spike train
        """
        # Encode
        # Normalize x to [0, 1] for probability
        # Assuming input x is already somewhat normalized or z-scored
        # We apply a sigmoid or clip to get probabilities
        
        prob = 1 / (1 + np.exp(-(self.gain * x + self.offset)))
        spikes = torch.rand_like(torch.tensor(prob, dtype=torch.float32)) < torch.tensor(prob, dtype=torch.float32)
        return spikes.float()

    def _latency_coding(self, x):
        """
        Latency coding: Stronger input -> Earlier spike.
        Not typically used for continuous time-series in this exact way,
        usually for static images converted to time.
        For now, we will stick to rate coding for timecourses.
        """
        raise NotImplementedError("Latency coding not yet implemented for timecourses")
