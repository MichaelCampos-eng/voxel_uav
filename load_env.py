import gymnasium as gym
from stable_baselines3 import A2C
from environment import Drone_Env

env = Drone_Env()

models_dir = "models/A2C"
model_path = f"{models_dir}/90000.zip"

model = A2C.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

env.close()