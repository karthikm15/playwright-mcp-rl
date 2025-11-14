"""Behavior cloning trainer."""

class BCTrainer:
    """Supervised learning on expert demonstrations."""
    
    def __init__(self, policy, config):
        self.policy = policy
        self.config = config
    
    def train(self, demos):
        """Train policy on expert trajectories."""
        # Placeholder: load demos, compute loss, update policy
        # for demo in demos:
        #     loss = cross_entropy(policy(demo.obs), demo.actions)
        #     loss.backward()
        pass

