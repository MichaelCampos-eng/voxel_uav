from stable_baselines3.common.env_checker import check_env
from environment import Drone_Env

env = Drone_Env()
episodes = 50

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        print("action", action)
        obs, reward, done, truncated, info = env.step(action)
        print('reward', reward)