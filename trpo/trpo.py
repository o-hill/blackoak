'''

    Trust Region Policy Optimization Agent.

'''

import numpy as np
import tensorflow as tf

from utils import FlattenTheta, IntegrateTheta


class Agent:

    def __init__(self, env, pdf, name: str, load_model: bool) -> None:
        '''Initialize the TRPO agent with an environment and pdf.'''

        self.env = env
        self.pdf = pdf
        self.name = name
        self.session = tf.Session()

        # Do we already have networks? self.policy is defined after this step.
        self.load() if load_model else self.initialize_networks()

        # Extract and replace updated weights into the policy network.
        self.flatten_theta = FlattenTheta(self.session, self.policy.trainable_weights)
        self.integrate_theta = IntegrateTheta(self.session, self.policy.trainable_weights)


    def single_path(self) -> None:
        '''Perform a single path rollout.'''

        # Randomly sample the initial state.
        state = self.env.reset()
        from ipdb import set_trace as debug; debug() # LOOK AT THE STATE OBJECT.
        done = False

        # Stores (S_t, A_(t-1), AV_(t-1), R_t).
        history = [(state, -1, -1)]

        # Run a sample trajectory.
        while not done:

            # Evaluate the policy to find an action to take - A_(t-1).
            action_vectors, action = self.act(state)

            # Take the action!
            state, reward, done, info = self.env.step(action)
            history.append((state, action, action_vectors, reward))

        return np.array(history)


    def simulate_n_trajectories(self, n: int) -> np.array:
        '''Simulate N trajectories in the environment.'''

        trajectories = [ ]
        for _ in range(n):
            trajectories.append(self.single_path())

        return np.array(trajectories)


    def learn(self, n_trajectories: int) -> None:
        '''Learn an optimal policy given an environment.'''

        # Simulate a series of trajectories in the environment.
        trajectories = self.simulate_n_trajectories(n_trajectories)

        # Compile the necessary information for the TRPO update.
        from ipdb import set_trace as debug; debug() # VALIDATE STATES INDEXING.
        states = trajectories[:, :, 0]
        actions = trajectories[:, :, 1]
        action_vectors = trajectories[:, :, 2]
        rewards = trajectories[:, :, 3]

        # TRPO update!
        theta = self.flatten_theta()

        # Compute the policy gradient -> Fx = g.
        g = self.session.run(self.policy_gradient, feed_dict = {
            self.action_vectors: action_vectors,
            self.actions: actions,
            self.states: states
        })

        # Evaluate the natural gradient using conjugate gradient descent.
        natural_gradient = conjugate_gradient_descent(fisher_vp, g)



















