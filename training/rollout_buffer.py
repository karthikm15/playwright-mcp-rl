"""Rollout storage buffer."""

class RolloutBuffer:
    """Stores trajectories for training."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, obs, action, reward, value, log_prob, done):
        """Add transition to buffer."""
        # Placeholder: append to lists
        pass
    
    def clear(self):
        """Clear buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def get_batch(self):
        """Return batched data for training."""
        # Placeholder: return tensors
        return None

