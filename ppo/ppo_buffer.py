'''

    A class to store the trajectory information for PPO.

'''


import numpy as np
import scipy.signal



def combined_shape(size: int, shape) -> tuple:
    '''Return a tuple of the size and the shape, which could be a scalar or a tuple itself.'''
    return (size, shape) if np.isscalar(shape) else (size, *shape)


def discounted_cumsum(x, discount):
    """
    TAKEN FROM https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/tf1/ppo/core.py
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



class Storage:
    '''Stores trajectories experienced by an agent.'''

    def __init__(self, observations_dim: int, action_dim: int, capacity: int, gamma: float = 0.99, lambda_: float = 0.95) -> None:
        '''Initialize storage for all the necessary components.'''

        self.observations = np.zeros(combined_shape(capacity, observations_dim), dtype=np.float32)
        self.actions = np.zeros(combined_shape(capacity, action_dim), dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.logp = np.zeros(capacity, np.float32)
        self.gamma, self.lambda_ = gamma, lambda_
        self.current = 0
        self.trajectory_start = 0
        self.capacity = capacity


    def push(self, observation, action, reward: float, value: float, logp: float) -> None:
        '''Add a step to storage.'''

        if not self.current < self.capacity:
            raise RuntimeError('No space left in storage!')

        self.observations[self.current] = observation
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.values[self.current] = value
        self.logp[self.current] = logp

        self.current += 1


    def end_trajectory(self, final_value: float) -> None:
        '''End the current trajectory and compute advantages and returns for use in training.'''

        rewards = np.append(self.rewards[self.trajectory_start : self.current], final_value)
        values = np.append(self.values[self.trajectory_start : self.current], final_value)

        # Compute advantage estimation.
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[self.trajectory_start : self.current] = discounted_cumsum(deltas, self.gamma * self.lambda_)

        # Compute returns for targets in the value function.
        self.returns[self.trajectory_start : self.current] = discounted_cumsum(rewards, self.gamma)[:-1]

        # Start a new trajectory.
        self.trajectory_start = self.current


    def get(self) -> tuple:
        '''Get both the recorded and computed data.'''

        if self.current != self.capacity:
            raise RuntimeError(f'The epoch is not over yet! ({self.current}/{self.capacity})')

        # Reset indices.
        self.current, self.trajectory_start = 0, 0

        # Normalize the advantage estimates.
        mu, sigma = self.advantages.mean(), self.advantages.std()
        self.advantages = (self.advantages - mu) / sigma

        return (
            self.observations,
            self.actions,
            self.advantages,
            self.returns,
            self.logp,
        )


