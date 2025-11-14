"""Evaluation script."""

import sys
sys.path.append('..')

from env.browser_env import BrowserEnv
from models.policy import PolicyNetwork

def evaluate(policy, test_tasks):
    """Run policy on test tasks and record success rate."""
    # Placeholder: load policy, run on test tasks
    # successes = 0
    # for task in test_tasks:
    #     env = BrowserEnv(task)
    #     obs = env.reset()
    #     done = False
    #     while not done:
    #         action = policy.act(obs)
    #         obs, reward, done, _ = env.step(action)
    #     if reward > 0:  # success condition
    #         successes += 1
    # return successes / len(test_tasks)
    return 0.0

if __name__ == '__main__':
    # Placeholder: load policy, test_tasks, run evaluation
    pass

