import gymnasium as gym
from stable_baselines3 import A2C, PPO
from environment import Drone_Env
import argparse
import os

def save_model(models_dir, algo, policy, log_name):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists("logs"):
        os.makedirs("logs")

    env = Drone_Env()
    model = algo(policy, env, verbose=1, tensorboard_log="logs")

    TIMESTEPS = 10000
    for i in range(1, 100000):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=log_name)
        model.save(f"{models_dir}/{TIMESTEPS * i}")

    env.close()

# Example command: python3 save_model.py -n "A2C" -a "A2C" -p "MultiInputPolicy"
if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Train and save models.")

    # Add arguments
    parser.add_argument("-n", "--name", type=str, help="Directory to save models.")
    parser.add_argument("-a", "--algorithm", type=str, help="Type of algo to train with like PPO.")
    parser.add_argument("-p", "--policy", type=str, help="Policy for model.")

    # Parse the arguments
    args = parser.parse_args()

    policy = "MultiInputPolicy" if args.policy == None else args.policy
    models_dir = f"models/{args.name}"
    algo = None
    if args.algorithm == "PPO":
        algo = PPO
    elif args.algorithm == "A2C":
        algo = A2C

    save_model(models_dir, algo, policy, args.algorithm)
