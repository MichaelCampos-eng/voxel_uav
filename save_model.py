import gymnasium as gym
from stable_baselines3 import A2C
from environment import Drone_Env
import os

def save_model(models_dir, logdir, model_type):
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = Drone_Env()
    model = model_type("MultiInputPolicy", env, verbose=1, tensorboard_log="logs")

    TIMESTEPS = 10000
    for i in range(1, 10000000):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="AC2")
        model.save(f"{models_dir}/{TIMESTEPS * i}")

    env.close()

if __name__ == '__main__':
    save_model(models_dir="models/A2C", logdir="logs", model_type=A2C)
