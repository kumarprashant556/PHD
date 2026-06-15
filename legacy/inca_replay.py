import random
import torch

class ExperienceReplay:
    """
    Reservoir Sampling Buffer.
    Maintains a fixed-size memory of past examples to prevent forgetting
    within the current active block.
    """
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        
    def add(self, batch_data):
        """
        Add new examples (text strings) to buffer.
        If full, random replacement (Reservoir Sampling).
        """
        # batch_data is a list of text strings or dicts
        for item in batch_data:
            if len(self.buffer) < self.capacity:
                self.buffer.append(item)
            else:
                # Randomly replace an existing item
                idx = random.randint(0, self.capacity - 1)
                self.buffer[idx] = item
                
    def sample(self, batch_size):
        """Returns a list of samples."""
        if len(self.buffer) < batch_size:
            return self.buffer # Return all if not enough
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)