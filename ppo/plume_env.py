'''

    A simulation environment in which an agent discovers information about
    multivariate gaussian plumes in three dimensions.

'''


import numpy as np
import scipy.stats
import gym
import fastkde

from matplotlib import pyplot as plt


class Agent:
    '''The environments understanding of an agent.'''

    def __init__(self, position: np.ndarray, boundary) -> None:
        '''Initialize the agent.'''
        
        # Position: [x, y].
        self.position = position
        self.states_visited = set([tuple(self.position)])
        self.path = [self.position]
        self.boundary = boundary


    def move(self, action: int) -> None:
        '''Move the agents position.'''

        if action == 0: # Left
            self.position[1] = max(0, self.position[1] - 1)
        elif action == 1: # Right
            self.position[1] = min(self.boundary - 1, self.position[1] + 1)
        elif action == 2: # Up
            self.position[0] = max(0, self.position[0] - 1)
        elif action == 3: # Down
            self.position[0] = min(self.boundary - 1, self.position[0] + 1)

        # Mark this state as visited and add it to the path.
        self.states_visited.add(tuple(self.position))
        self.path.append(list(self.position))


    def position_variance(self) -> float:
        '''Find the variance in the agents position.'''
        path_array = np.array(self.path)
        return np.sqrt(path_array[:, 0].std() ** 2 + path_array[:, 1].std() ** 2)




class Plume(gym.Env):


    def __init__(self, edge_length: int = 120) -> None:
        '''Set up the environment.'''

        # Set up some basic parameters.
        self.edge_length = edge_length
        self.background_concentration = 350
        self.gaussian_mean = 100
        self.info = { }
        self.agent = Agent(np.random.randint(edge_length, size=2), self.edge_length)
        self.max_position_variance = self.get_maximum_variance()

        # The true distributions.
        self.means = [
            [60, 85],
            [16, 48],
            [96, 18]
        ]

        self.covariances = [
            [[670, 0], [40, 770]],
            [[575, 100.0], [23.0, 850]],
            [[570.0, 100.0], [-100.0, 940.5]],
        ]

        # Up, down, left, right.
        self.action_space = gym.spaces.Discrete(4)

        # Two-dimensional position, plume concentration.
        self.observation_space = gym.spaces.Box(np.array((0, 0, -np.inf)), np.array((1, 1, np.inf)))

        # Create the grid in memory.
        x, y = np.linspace(0, 119, 120, dtype=int), np.linspace(0, 119, 120, dtype=int)
        self.X, self.Y = np.meshgrid(x, y)

        # Get distributions.
        self.distributions = [
            scipy.stats.multivariate_normal(mean=m, cov=c)
            for m, c in zip(self.means, self.covariances)
        ]

        # Find the mean value for each distribution.
        self.mean_values = [
            dist.pdf(m) for m, dist in zip(self.means, self.distributions)
        ]

        # Compute and plot the concentration matrix.
        self.concentration_matrix = self.get_concentration_matrix()
        plt.subplot(1, 2, 1)
        plt.imshow(self.concentration_matrix)

        self.ax = plt.subplot(1, 2, 2)
        self.ax.imshow(self.concentration_matrix)


    def step(self, action: int) -> tuple:
        '''Take an action in the environment.'''
        self.agent.move(action)
        return self.observation, self.reward, self.done, self.info


    def reset(self) -> np.ndarray:
        '''Reset the environment.'''
        self.agent = Agent(np.random.randint(self.edge_length, size=2), self.edge_length)
        return self.observation


    def render(self) -> None:
        '''Render the environment, and the agents path.'''

        # Plot the agents path over the already shown matrix.
        self.ax.scatter(self.agent.path[-1][0], self.agent.path[-1][1], c='b')
        plt.show()



    @property
    def observation(self) -> np.ndarray:
        '''An array of [agent.position.x, agent.position.y, concentration].'''
        return np.array([
            *(self.agent.position / self.edge_length),
            self.concentration(self.agent.position[0], self.agent.position[1])
        ])


    @property
    def reward(self) -> float:
        '''Reward the agent with concentration plus the variance of its position.'''
        concentration = self.concentration(self.agent.position[0], self.agent.position[1])
        position_variance = self.agent.position_variance()

        reward = concentration + position_variance

        return reward


    @property
    def done(self) -> bool:
        '''Has the agent reached a terminal state?'''
        return len(self.agent.states_visited) >= self.edge_length ** 2 or len(self.agent.path) >= 1200


    def concentration(self, x: np.ndarray, y: np.ndarray) -> float:
        '''Get the concentration at a point.'''
        concen = sum([
            self.gaussian_mean * (dist.pdf(np.dstack((x, y))) / mu_dist)
            for dist, mu_dist in zip(self.distributions, self.mean_values)
        ])

        return concen


    def scale_concentration(self, concentration: float) -> float:
        '''Scale the concentration for the reward.'''
        return (concentration - self.background_concentration) / self.gaussian_mean


    def get_concentration_matrix(self) -> None:
        '''Get the distributions for plotting.'''
        return self.concentration(self.X.ravel(), self.Y.ravel()).reshape(self.X.shape)


    def get_maximum_variance(self) -> float:
        '''Find the maximum position variance for the agent.'''

        fake_agent = Agent([0, 0], self.edge_length)
        max_variance_path = [ ]

        # Assemble the path that will create the most variance (a ring around the edge).
        for i in np.linspace(0, self.edge_length - 1, self.edge_length, dtype=int):
            max_variance_path.append([i, 0])
            max_variance_path.append([i, 119])
            max_variance_path.append([0, i])
            max_variance_path.append([119, i])

        # Compute the variance of the path.
        fake_agent.path = max_variance_path
        return fake_agent.position_variance()


if __name__ == '__main__':

    env = Plume()
    env.render()




