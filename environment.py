import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame


class Drone_Env(gym.Env):
    """Custom Environment that follows gym interface."""
    metadata = {"render_modes": ["human"], "render_fps": 7}
    window_size = 512
    grid_size = 10
    window = None
    clock = None

    global_step = 100

    bounds = np.array([grid_size, grid_size])
    action_to_direction = {
        0: np.array([1, 0]),  # right
        1: np.array([0, 1]),  # up
        2: np.array([-1, 0]),  # left
        3: np.array([0, -1])  # down
    }

    def __init__(self, render_mode="human"):
        super().__init__()
        np.random.seed(42)
        self.desired_traj = Drone_Env.generate_lawnmower_trajectory(self.grid_size)
        self.render_mode = render_mode
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.MultiDiscrete([4, 3])
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=int)})
        self.reset()

    def step(self, action):
        """
        1) Get current target and find if its currently on an obstacle, if so update the target
        2) Update the current state given action
        3) If the agent reached the target modified, update self.index
        """
        reward_traj = 0
        reward_obs = 0
        reward_life = 0

        # If we complete the pathway, we finish the game
        if self.index >= len(self.desired_traj):
            self.done = True
            reward = 100
            return {"agent": self.agent_state, "target": self.target_state}, reward, self.done, False, {}
        else:
            self.target_state = self.desired_traj[self.index]

        # Update the grid we are working with
        self.global_time = self.global_time % self.global_step
        if self.global_time == 0:
            grid = Drone_Env.generate_random_matrix(self.grid_size, self.grid_size)
            self.grid, self.obstacle_locations = Drone_Env.update_grid(grid, self.agent_state)
        self.global_time += 1

        # Get feasible target
        self.target_state = self.next_best_point(self.agent_state, self.target_state)
        direction, speed = action[0], action[1]
        move = self.action_to_direction[direction] * speed

        # Update agent
        self.agent_state += move

        # If we reach the target position, we increase reward
        if np.all(self.agent_state == self.target_state):
            self.index += 1
            reward_traj += 10
            self.target_state = self.desired_traj[self.index]

        # Quit the game if we go out of bounds
        if self.agent_state[0] < 0 or self.agent_state[0] > self.grid_size - 1 or \
            self.agent_state[1] < 0 or self.agent_state[1] > self.grid_size - 1:
            self.done = True
            reward = -100
            return {"agent": self.agent_state, "target": self.target_state}, reward, self.done, False, {}

        r, c = self.agent_state[0], self.agent_state[1]
        if self.grid[r, c] == 1:
            self.done = True

        reward_traj -= Drone_Env.manhattan_distance(self.agent_state, self.target_state)
        reward_obs -= 10/(1 + Drone_Env.closest_manhattan_distance(self.obstacle_locations, self.agent_state))
        reward_life -= 1
        reward = reward_traj + reward_obs + reward_life

        return {"agent": self.agent_state, "target": self.target_state}, reward, self.done, False, {}

    def reset(self, seed=None, options=None):
        self.global_time = 0
        self.done = False
        self.index = 1
        self.agent_state = np.array([0, 0])
        self.target_state = self.desired_traj[self.index]
        grid = Drone_Env.generate_random_matrix(self.grid_size, self.grid_size)
        self.grid, self.obstacle_locations = Drone_Env.update_grid(grid, self.agent_state)
        return {"agent": self.agent_state, "target": self.target_state}, {}

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((140, 207, 127))
        pix_square_size = (self.window_size / self.grid_size)

        for obstacle in self.obstacle_locations:
            pygame.draw.circle(
                canvas,
                (0, 82, 33),
                (obstacle + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (222, 232, 222),
            (self.agent_state + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Now we draw the target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self.target_state + 0.5) * pix_square_size,
            pix_square_size / 4,
        )

        # Finally, add some gridlines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def next_best_point(self, agent, target):
        for increment in range(self.grid_size):
            best = np.array([None])
            best_distance = np.Infinity
            for i in range(-increment, increment + 1):
                for j in range(-increment, increment + 1):
                    explore = np.array([target[0] + i, target[1] + i])
                    explore = np.clip(explore, 0, self.grid_size - 1)
                    heuristic_te = Drone_Env.manhattan_distance(np.array([target]), np.array([explore]))[0]
                    heuristic_ae = Drone_Env.manhattan_distance(np.array([agent]), np.array([explore]))[0]
                    if self.grid[explore[0], explore[1]] == 0:
                        distance = heuristic_te + heuristic_ae
                        if best_distance > distance:
                            best = explore
                            best_distance = distance
            if best.any() != None:
                return best
        return None

    @staticmethod
    def generate_lawnmower_trajectory(grid_size):
        trajectory = []
        for row in range(grid_size):
            if row % 2 == 0:
                for col in range(grid_size):
                    trajectory.append(np.array([row, col]))
            else:
                for col in range(grid_size - 1, -1, -1):
                    trajectory.append(np.array([row, col]))
        return trajectory

    @staticmethod
    def generate_random_matrix(rows, cols, prob_of_ones=0.01):
        if not 0 <= prob_of_ones <= 1:
            raise ValueError("Probability of ones must be between 0 and 1.")
        matrix = np.random.choice([1, 0], size=(rows, cols), p=[prob_of_ones, 1 - prob_of_ones])
        return matrix

    @staticmethod
    def update_grid(grid, anchor, space=1):
        _, grid_size = grid.shape
        safe = np.array([anchor - space, anchor + space]).T
        safe = np.clip(safe, 0, grid_size - 1)
        rows, cols = safe
        grid[rows[0]: rows[1] + 1, cols[0]: cols[1] + 1] = 0
        obstacle_locations = np.array(np.where(grid == 1)).T
        return grid, obstacle_locations

    @staticmethod
    def manhattan_distance(p1, p2):
        if p1.shape == (2,):
            return np.sum(np.abs(p1 - p2))
        return np.sum(np.abs(p1 - p2), axis=1)

    @staticmethod
    def closest_manhattan_distance(indices, point):
        if indices.size == 0:
            return 0
        point_tiled = np.tile(point, (indices.shape[0], 1))
        return np.min(Drone_Env.manhattan_distance(indices, point_tiled))