'''

    Implementation of the actor-critic networks and policies, based on OpenAI/PPO.

'''


import tensorflow as tf
import numpy as np
from gym.spaces import Box, Discrete



epsilon = 1e-8


# -----------------------------------------------------------------------------
#                               Utilities
# -----------------------------------------------------------------------------


def gaussian_likelihood(x, mu, log_sigma) -> float:
    '''Gaussian likelihood of a vector X in the distribution N(mu, exp(log_sigma)**2).'''

    left = -0.5 * ((x - mu) / (tf.exp(log_sigma) + epsilon)) ** 2
    right = 2 * log_sigma + np.log(2 * np.pi)

    return tf.reduce_sum(left + right, axis=1)


def get_space_shape(space) -> int:
    '''Determine the shape of the action space.'''

    if isinstance(space, Box):
        if len(space.shape) > 1:
            raise RuntimeError('Space dimension is more than 1!')

        return space.shape[0]

    # Actually just an integer.
    elif isinstance(space, Discrete):
        return space.n

    else:
        raise RuntimeError('Environments space is incompatible.')



# -----------------------------------------------------------------------------
#                               Policies
# -----------------------------------------------------------------------------


class MLPPolicy(tf.Module):

    def __init__(self,
            action_dim: int,
            observation_dim,
            hidden_sizes: tuple,
            vf_lr: float,
            pi_lr: float,
            activation,
            output_activation,
            clip_ratio) -> None:
        '''Define a policy.'''

        self.log_prob_old = None
        self.action_dim = action_dim
        self.feature_extraction = mlp(observation_dim, list(hidden_sizes[:2]))
        self.loss_fn = None
        self.clip_ratio = clip_ratio

        self.policy_network = mlp(hidden_sizes[1], list(hidden_sizes[2:]) + [self.action_dim], output_activation=output_activation)
        self.value_network = mlp(hidden_sizes[1], list(hidden_sizes[2:]) + [1], activation=activation)

        self.value_optimizer = tf.keras.optimizers.Adam(vf_lr)
        self.policy_optimizer = tf.keras.optimizers.Adam(pi_lr)


    @tf.function
    def train_step_policy(self,
            observations: np.ndarray,
            advantages: np.ndarray,
            actions: np.ndarray,
            log_prob_old: np.ndarray,
            train_pi_iters: int,
            target_kl: float) -> None:
        '''Train the policy network.'''

        # Train the policy network.
        for i in tf.range(train_pi_iters):
            with tf.GradientTape() as tape:
                pi_loss, approximate_kl, _ = self.policy_loss(observations, advantages, actions, log_prob_old)

            trainable_variables = self.policy_network.trainable_variables + self.feature_extraction.trainable_variables
            gradients = tape.gradient(pi_loss, trainable_variables)
            self.policy_optimizer.apply_gradients(zip(gradients, trainable_variables))

            if approximate_kl > 1.5 * target_kl:
                print(f'Stopping policy training at step {i}/{train_pi_iters}.')
                break


    @tf.function
    def train_step_value(self,
            observations: np.ndarray,
            returns: np.ndarray,
            train_value_iters: int) -> None:
        '''Train the value network.'''

        for i in tf.range(train_value_iters):
            with tf.GradientTape() as tape:
                value_loss = self.value_loss(observations, returns)

            trainable_variables = self.value_network.trainable_variables + self.feature_extraction.trainable_variables
            gradients = tape.gradient(value_loss, trainable_variables)
            self.value_optimizer.apply_gradients(zip(gradients, trainable_variables))


    @tf.function
    def policy_loss(self,
            observations: np.ndarray,
            advantages: np.ndarray,
            actions: np.ndarray,
            log_prob_old: np.ndarray) -> tuple:
        '''Compute loss for pi and value, as well as entropy.'''

        pi, log_prob, log_prob_pi = self.get_objectives(observations, actions)
        pi_ratio = tf.exp(log_prob - log_prob_old)
        min_advantage = tf.where(
            advantages > 0,
            (1 + self.clip_ratio) * advantages,
            (1 - self.clip_ratio) * advantages
        )

        pi_loss = -tf.reduce_mean(tf.minimum(pi_ratio * advantages, min_advantage))

        approximate_kl = tf.reduce_mean(log_prob_old - log_prob)
        approx_ent = tf.reduce_mean(-log_prob)

        return pi_loss, approximate_kl, approx_ent


    @tf.function
    def value_loss(self, observations, returns) -> float:
        '''Calculate value network loss.'''
        value_prediction = self.value(observations)
        return tf.reduce_mean((returns - value_prediction) ** 2)


    def estimate(self, observation, action) -> tuple:
        '''Estimate the policy.'''
        raise NotImplementedError


    @tf.function
    def policy(self, observation) -> np.ndarray:
        '''Predict the mean of the action distribution, given an observation.'''
        features = self.feature_extraction(observation)
        return self.policy_network(features)


    @tf.function
    def value(self, observation) -> float:
        '''Calculate the predicted discounted reward of this state.'''
        features = self.feature_extraction(observation)
        return self.value_network(features)



class MLPCategoricalPolicy(MLPPolicy):

    def __init__(
            self,
            observation_dim,
            action_shape,
            hidden_sizes,
            activation,
            output_activation,
            clip_ratio,
            pi_lr,
            vf_lr) -> tuple:
        '''Define a discrete policy.'''

        # Create a network that takes in observations and outputs actions.
        super().__init__(action_shape, observation_dim, hidden_sizes, vf_lr, pi_lr, activation, output_activation, clip_ratio)


    @tf.function
    def observe(self, observation, get_log_prob_all: bool = False):
        '''Choose an action given an observation using the policy estimator.'''

        # Estimate pi(a|s)
        logits = self.policy(observation)

        self.log_prob_all = tf.nn.log_softmax(logits)
        self.pi = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        self.log_prob_pi = tf.reduce_sum(tf.one_hot(self.pi, depth=self.action_dim) * self.log_prob_all, axis=1)

        return (self.pi, self.log_prob_pi) if not get_log_prob_all else (self.pi, self.log_prob_pi, self.log_prob_all)


    @tf.function
    def get_objectives(self, observations, actions) -> tuple:
        '''Get the network objectives to optimize.'''

        pi, log_prob_pi, log_prob_all = self.estimate(observations, get_log_prob_all=True)
        log_prob = tf.reduce_sum(tf.one_hot(tf.cast(actions, dtype=tf.int32), depth=self.action_dim) * log_prob_all, axis=1)

        return pi, log_prob, log_prob_pi


class MLPGaussianPolicy(MLPPolicy):

    def __init__(
            self,
            observation_dim,
            action_shape,
            hidden_sizes,
            activation,
            output_activation,
            clip_ratio,
            pi_lr,
            vf_lr,) -> tuple:
        '''Define a continuous policy.'''

        # Create a network that takes in observations and outputs actions.
        super().__init__(action_shape, observation_dim, hidden_sizes, vf_lr, pi_lr, activation, output_activation, clip_ratio)
        self.log_sigma = tf.Variable(-0.5 * np.ones(self.action_dim, dtype=np.float32))


    @tf.function
    def observe(self, observation, actions = None) -> tuple:
        '''Choose an action given an observation using the policy estimator.'''

        # Estimate the mean and standard deviation of the distribution.
        mu = self.policy(observation)
        sigma = tf.exp(self.log_sigma)

        # Estimate pi using mu and sigma.
        self.pi = mu + tf.random.normal(tf.shape(mu)) * sigma

        if actions is not None:
            self.log_prob = gaussian_likelihood(actions, mu, self.log_sigma)

        self.log_prob_pi = gaussian_likelihood(self.pi, mu, self.log_sigma)

        return (self.pi, self.log_prob, self.log_prob_pi) if actions is not None else (self.pi, self.log_prob_pi)


    @tf.function
    def get_objectives(self, objectives, actions) -> tuple:
        '''Get the network objectives to optimize.'''
        return self.observe(objectives, actions)



# -----------------------------------------------------------------------------
#                               Networks
# -----------------------------------------------------------------------------


def mlp(input_shape, hidden_sizes, activation=tf.tanh, output_activation=None):
    '''Define a multi-layer perceptron network.'''

    # Build the simple network.
    network = tf.keras.models.Sequential([
            tf.keras.layers.Dense(hidden_sizes[0], activation=activation, input_shape=[input_shape])
        ] + [
            tf.keras.layers.Dense(hs, activation=activation) for hs in hidden_sizes[1:-1]
        ] + [
            tf.keras.layers.Dense(hidden_sizes[-1], activation=output_activation)
    ])

    return network


def mlp_actor_critic(env, clip_ratio, pi_lr, vf_lr, hidden_sizes: tuple = (32, 64, 64, 64)) -> tuple:
    '''Return network outputs for different variables.'''

    activation = tf.tanh
    output_activation = None
    observation_dim = get_space_shape(env.observation_space)
    action_shape = get_space_shape(env.action_space)
    
    # Define the policy based on the problem (discrete or continuous).
    if isinstance(env.action_space, Box):
        chosen_policy = lambda *args, **kwargs: MLPGaussianPolicy(*args, **kwargs)

    elif isinstance(env.action_space, Discrete):
        chosen_policy = lambda *args, **kwargs: MLPCategoricalPolicy(*args, **kwargs)

    policy = chosen_policy(
        observation_dim = observation_dim,
        action_shape = action_shape,
        hidden_sizes = hidden_sizes,
        activation = activation,
        output_activation = output_activation,
        clip_ratio = clip_ratio,
        pi_lr = pi_lr,
        vf_lr = vf_lr,
    )

    return policy