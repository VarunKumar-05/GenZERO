import torch
import numpy as np

class SpikeEncoder:
    """
    Converts continuous time-series data into discrete spike trains.
    """
    def __init__(self, method='rate', gain=1.0, offset=0.0, threshold=0.5):
        self.method = method
        self.gain = gain
        self.offset = offset
        self.threshold = threshold

    def __call__(self, x, seed=None):
        if self.method == 'rate':
            return self._rate_coding(x, seed)
        elif self.method == 'threshold':
            return self._threshold_coding(x)
        elif self.method == 'delta':
            return self._delta_coding(x)
        elif self.method == 'latency':
            return self._latency_coding(x)
        else:
            raise ValueError(f"Unknown encoding method: {self.method}")

    def _rate_coding(self, x, seed=None):
        """
        Rate coding: Probability of firing is proportional to intensity.
        Input x: (Time, Channels)
        Output: (Time, Channels) - Binary spike train
        
        If 'seed' is provided, ensures deterministic output.
        """
        # Normalize/Scale to likelihood
        prob = 1 / (1 + np.exp(-(self.gain * x + self.offset)))
        prob_tensor = torch.tensor(prob, dtype=torch.float32)
        
        if seed is not None:
            # Create a local generator to avoid affecting global state
            gen = torch.Generator()
            gen.manual_seed(int(seed))
            rand_mask = torch.rand(prob_tensor.shape, generator=gen)
        else:
            rand_mask = torch.rand_like(prob_tensor)
            
        spikes = rand_mask < prob_tensor
        return spikes.float()

    def _threshold_coding(self, x):
        """
        Deterministic: Spike if value > threshold.
        """
        spikes = torch.tensor(x > self.threshold, dtype=torch.float32)
        return spikes

    def _delta_coding(self, x):
        """
        Spike if change in signal |x_t - x_{t-1}| > threshold.
        """
        x_tensor = torch.tensor(x, dtype=torch.float32)
        diff = torch.zeros_like(x_tensor)
        diff[1:] = torch.abs(x_tensor[1:] - x_tensor[:-1])
        spikes = (diff > self.threshold).float()
        return spikes

    def _latency_coding(self, x):
        """
        Latency coding: Stronger input -> Earlier spike.
        Not typically used for continuous time-series in this exact way,
        usually for static images converted to time.
        For now, we will stick to rate coding for timecourses.
        """
        raise NotImplementedError("Latency coding not yet implemented for timecourses")
