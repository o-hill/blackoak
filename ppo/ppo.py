'''

    PPO agent implementation.

'''

from ppo_buffer import Storage
from actor_critic import mlp_actor_critic, get_space_shape

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from time import sleep


class PPOAgent:

    def __init__(
            self,
            env_function,
            actor_critic = mlp_actor_critic,
            seed: int = 0,
            steps_per_epoch: int = 500,
            n_epochs: int = 50,
            gamma: float = 0.99,
            clip_ratio: float = 0.2,
            pi_lr: float = 3e-4,
            vf_lr: float = 1e-3,
            train_pi_iters: int = 80,
            train_value_iters: int = 80,
            lambda_: float = 0.97,
            max_episode_length: int = 1000,
            target_kl: float = 0.01,
            save_frequency: int = 10) -> None:
        '''Initialize the agent.'''

        self.__dict__.update(locals())

        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.env = env_function()
        observation_dim = get_space_shape(self.env.observation_space)
        action_dim = get_space_shape(self.env.action_space)

        # Set up policy and value networks.
        self.policy = actor_critic(
            env = self.env,
            clip_ratio = clip_ratio,
            pi_lr = pi_lr,
            vf_lr = vf_lr,
        )

        # Keep experience in memory.
        self.storage = Storage(
            observations_dim = observation_dim,
            action_dim = action_dim,
            capacity = steps_per_epoch,
        )


    def run(self) -> None:
        '''Collect experience and train the policy and value networks.'''

        observation = self.env.reset()
        reward = 0
        done = False
        ep_return = 0
        ep_length = 0

        for epoch in range(self.n_epochs):
            for t in tqdm(range(self.steps_per_epoch)):

                action_t, log_prob_t = self.policy.observe(np.array([observation]))
                value_t = self.policy.value(np.array([observation]))

                action_t, log_prob_t, value_t = action_t.numpy(), log_prob_t.numpy(), value_t.numpy()

                self.env.render()
                observation_t, reward, done, _ = self.env.step(action_t[0])
                ep_return += reward
                ep_length += 1

                # Store the experience for future updates.
                self.storage.push(
                    observation = observation,
                    action = action_t,
                    reward = reward,
                    value = value_t,
                    logp = log_prob_t,
                )

                observation = observation_t
                terminal = done or (ep_length == self.max_episode_length)

                # End the trajectory if we are done.
                if terminal or t == self.steps_per_epoch - 1:
                    if not terminal:
                        print(f'Trajectory cut off before terminal state at {ep_length}.')

                    # Determine the final value estimate.
                    last_value = 0 if done else self.policy.value(np.array([observation]))
                    self.storage.end_trajectory(last_value)

                    # Reset the environment and prepare for the new trajectory.
                    observation, ep_return, ep_length = self.env.reset(), 0, 0

            # Update the models.
            print(f'\n------- Network Training at epoch: {epoch}/{self.n_epochs}. --------')
            self.update_networks()

            # Save the model.
            if epoch % self.save_frequency == 0 or epoch == self.n_epochs - 1:
                self.policy.policy_network.save('./networks/lunar_policy.h5')
                self.policy.value_network.save('./networks/lunar_value.h5')
                self.policy.feature_extraction.save('./networks/lunar_features.h5')


    def update_networks(self) -> None:
        '''Update the policy and value networks.'''

        observations, actions, advantages, returns, log_probs, rewards = self.storage.get()
        pi_loss_old, value_loss_old, entropy_old = self.calculate_loss(observations, advantages, returns, actions, log_probs)

        print(f'Average reward: {rewards.mean()}')

        # Train the policy network.
        self.policy.train_step_policy(
            observations = observations,
            advantages = advantages,
            actions = actions,
            log_prob_old = log_probs,
            train_pi_iters = self.train_pi_iters,
            target_kl = self.target_kl,
        )

        # Train the value network.
        self.policy.train_step_value(
            observations = observations,
            returns = returns,
            train_value_iters = self.train_value_iters,
        )

        # Re-evaluate loss.
        pi_loss_new, value_loss_new, entropy_new = self.calculate_loss(observations, advantages, returns, actions, log_probs)

        print('----------------------')
        print(f'Old pi loss: {pi_loss_old}')
        print(f'New pi loss: {pi_loss_new}')
        print(f'Delta pi loss: {abs(pi_loss_new - pi_loss_old)}')
        print('----------------------')
        print(f'Old value loss: {value_loss_old}')
        print(f'New value loss: {value_loss_new}')
        print(f'Delta value loss: {abs(value_loss_new - value_loss_old)}')
        print('----------------------\n\n')


    def calculate_loss(self, observations, advantages, returns, actions, log_probs) -> tuple:
        '''Calculate the loss of the policy and value networks.'''

        # Evaluate the policy network loss.
        pi_loss, _, entropy = self.policy.policy_loss(observations, advantages, actions, log_probs)

        # Evaluate the value network loss.
        value_loss = self.policy.value_loss(observations, returns)

        return pi_loss, value_loss, entropy



if __name__ == '__main__':

    # from beast_gym import Beast
    # from plume_env import Plume
    # agent = PPOAgent(lambda: Plume())
    # agent.run()

    # env = Plume()
    # obs = env.reset()
    # for t in range(1200):

    #     action, _ = agent.policy.observe(np.array([obs]))
    #     action = action.numpy()
    #     obs, rew, done, info = env.step(action)

    # from matplotlib import pyplot as plt
    # plt.imshow(env.concentration_matrix)

    # path_array = np.array(env.agent.path)
    # plt.scatter(path_array[:, 0], path_array[:, 1], s=5, c='b')
    # plt.scatter(path_array[-1, 0], path_array[-1, 1], s=5, c='r')
    # plt.show()


        


    import gym
    agent = PPOAgent(lambda: gym.make('LunarLanderContinuous-v2'))
    agent.run()
