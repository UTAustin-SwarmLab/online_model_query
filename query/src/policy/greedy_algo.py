import argparse

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3.common.utils import set_random_seed

import query.envs  # noqa: F401


class egreedy:
    def __init__(self, num_arms):  ## Initialization
        self.num_arms = num_arms
        self.C = 1
        self.delta = 0.1
        self.restart()
        return None

    def restart(
        self,
    ):
        self.time = 0.0
        self.eps = 1

        ## Your code here
        self.emp_means = np.zeros(self.num_arms)
        self.num_pulls = np.zeros(self.num_arms)

        return None

    def get_best_arm(
        self,
    ):
        if self.time < self.num_arms:
            arm = self.time % self.num_arms
        else:
            prob = np.random.uniform(0, 1)
            if prob < self.eps:
                arm = np.random.randint(0, self.num_arms)
            else:
                arm = np.argmax(self.emp_means)
        return int(arm)

    def update_stats(self, rew, arm):
        self.emp_means[arm] = (self.emp_means[arm] * self.num_pulls[arm] + rew) / (
            self.num_pulls[arm] + 1
        )
        self.num_pulls[arm] += 1
        self.time += 1
        self.eps = min(
            1, (self.num_arms * self.C) / ((self.time + 1) * (self.delta**2))
        )

        return None

    def iterate(self, arm, rew):
        self.update_stats(rew, arm)
        return


def run_algo(env, csv_path, num_iter):
    algo = egreedy(env.action_space.n)
    save_freq = 5
    cnt = 0
    total_reward = 0
    obs, info = env.reset()
    mean_reward_dict = {}
    for t in range(num_iter):
        cnt += 1
        action = algo.get_best_arm()
        # action = 2
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if cnt % save_freq == 0:
            mean_reward = total_reward / cnt
            mean_reward_dict[cnt] = mean_reward

        algo.iterate(action, reward)

    print("Saving mean reward dict...")
    # create an empty DataFrame
    df = pd.DataFrame(columns=["Step", "mean_reward"])
    df["Step"] = mean_reward_dict.keys()
    df["mean_reward"] = mean_reward_dict.values()
    df.to_csv(
        csv_path,
        index=False,
    )

    return


if __name__ == "__main__":
    ### poetry run python query/src/policy/greedy_algo.py -e OpenDomain -c True -n 5
    ### poetry run python query/src/policy/greedy_algo.py -e Alfred -c True -n 5
    ### poetry run python query/src/policy/greedy_algo.py -e Waymo -c True -n 5
    parser = argparse.ArgumentParser(description="Create Configuration")
    parser.add_argument("-e", "--env", type=str, help="environment name", default="")
    parser.add_argument(
        "-l", "--latency", type=bool, help="latency env or not", default=False
    )
    parser.add_argument(
        "-c", "--contextual", type=bool, help="comtxtual of not", default=False
    )
    parser.add_argument(
        "-i", "--return_image", type=bool, help="return image or not", default=False
    )
    parser.add_argument("-n", "--step", type=int, help="steps per update", default=100)
    args = parser.parse_args()
    print(args)

    # 'ImageNet1k', 'ImageNet1k_CIFAR100', 'ImageNet1k_CIFAR100Np', 'OpenBookQA', 'Alfred-v1'
    env_name = args.env + "Latency" if args.latency else args.env
    contextual = args.contextual  # False
    device = "cpu"
    max_episode_steps = args.step
    n_steps = args.step

    set_random_seed(42, using_cuda=device != "cpu")
    if "ImageNet" in env_name:
        env = gym.make(
            env_name + "-v1",
            max_episode_steps=max_episode_steps,
            device=device,
            p=[5 / 6, 1 / 6],
            return_image=args.return_image,
            contextual=contextual,
        )
    elif "QA" in env_name:
        answer = True
        total_timesteps = 50000
        print(env_name + "-v1")
        env = gym.make(
            env_name + "-v1",
            max_episode_steps=max_episode_steps,
            device=device,
            contextual=contextual,
            answer=answer,
            replace_sample=False,
        )
    elif "Domain" in env_name:
        total_timesteps = 12000
        answer = False
        # answer = True
        print(env_name + "-v1")
        env = gym.make(
            env_name + "-v1",
            max_episode_steps=max_episode_steps,
            device=device,
            contextual=contextual,
            answer=answer,
            replace_sample=False,
            save_freq=n_steps,
            save_reward=False,
        )
        tag = ""
        if contextual:
            tag += "_contextual"
        if answer:
            tag += "_answer"
    elif "Alfred" in env_name:
        print(env_name + "-v1")
        if "Latency" in env_name:
            reward_metric = "GC+PLW"
            env_name = env_name.replace("Latency", "")
        else:
            reward_metric = "SR"
        total_timesteps = 13000
        env = gym.make(
            env_name + "-v1",
            max_episode_steps=max_episode_steps,
            device=device,
            contextual=contextual,
            low_level=False,
            floor_plan=True,
            replace=False,
            reward_metric=reward_metric,
            save_freq=n_steps,
            save_reward=False,
        )
        env_name = "Alfred_" + reward_metric
    elif "Waymo" in env_name:
        total_timesteps = 20000
        env = gym.make(
            env_name + "-v1",
            max_episode_steps=max_episode_steps,
            device=device,
            contextual=contextual,
            text=True,
            replace=False,
            save_freq=n_steps,
            save_reward=False,
        )
    else:
        raise ValueError("env_name not found: " + env_name)
    csv_path = f"synced_data/cumulative_reward/{env_name}_greedy_step{n_steps}.csv"
    run_algo(env, csv_path, total_timesteps)
    env.close()

# Greedy Waymo
# Mean of acc list:  0.8360581970901455
# Mean of latency list:  -0.08461669667641855
# PPO Waymo
# Mean of acc list:  0.8793800258322569
# Mean of latency list:  -0.10567411209958148
# Greedy MMLU
# Mean of acc list:  0.7553537205232898
# Mean of latency list:  -0.26068982555245857
# PPO MMLU
# Mean of acc list:  0.7327963335879453
# Mean of latency list:  -0.13878054352958816
