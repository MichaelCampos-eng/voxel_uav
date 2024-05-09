import gymnasium as gym
from stable_baselines3 import A2C, PPO
from environment import Drone_Env
import argparse


def load_model(folder_name, load, algo, episodes=10):
    env = Drone_Env()

    models_dir = f"models/{folder_name}"
    model_path = f"{models_dir}/{load}"

    model = algo.load(model_path, env=env)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)

    env.close()

# Example command: python3 load_model.py -n "A2C" -a "A2C" -l "90000" -e 20
if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Train and save models.")

    # Add arguments
    parser.add_argument("-n", "--folder_name", type=str, help="Folder from which trained model was saved in.")
    parser.add_argument("-a", "--algorithm", type=str, help="Type of algo used to train with like PPO.")
    parser.add_argument("-l", "--load", type=str, help="The model name to check out.")
    parser.add_argument("-e", "--episodes", type=int, help="The model name to check out.")

    # Parse the arguments
    args = parser.parse_args()

    algo = None
    if args.algorithm == "PPO":
        algo = PPO
    elif args.algorithm == "A2C":
        algo = A2C
    load = f"{args.load}.zip"

    load_model(args.folder_name, load, algo, args.episodes)
