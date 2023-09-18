### tensorboard --logdir='./tensorboard_log/PPO_ImageNet1k' --port=6006
### python src/policy/ppo.py -e ImageNet1k_CIFAR100 -d 0 -c True -i True -n 100
### poetry run python query/src/policy/ppo.py -d 1 -e OpenBookQA -c True -n 100
### poetry run python query/src/policy/ppo.py -e OpenDomain -c True -n 100 
import argparse

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

import query.envs  # noqa: F401

parser = argparse.ArgumentParser(description="Create Configuration")
parser.add_argument("-e", "--env", type=str, help="environment name", default="")
parser.add_argument("-d", "--device", type=int, help="device name", default=-1)
parser.add_argument("-c", "--contextual", type=bool, help="comtxtual of not", default=False)
parser.add_argument("-i", "--return_image", type=bool, help="return image or not", default=False)
parser.add_argument("-n", "--step", type=int, help="steps per update", default=100)
args = parser.parse_args()
print(args)

# 'ImageNet1k', 'ImageNet1k_CIFAR100', 'ImageNet1k_CIFAR100Np', 'OpenBookQA'
env_name = args.env
contextual = args.contextual  # False
device = (
    f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
)
max_episode_steps = args.step
total_timesteps = 50000
n_steps = args.step  # 2048

if contextual:
    model_path = (
        f"./synced_data/models/{env_name}_step{n_steps}_PPO_img{args.return_image}.zip"
    )
else:
    model_path = f"./synced_data/models/{env_name}_step{n_steps}_PPO_non_contextual_img{args.return_image}.zip"

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
    log_path = f"./tensorboard_log/PPO_step{n_steps}_ImageNet1k_img{args.return_image}/"
elif "QA" in env_name:
    answer = True
    print(env_name + "-v1")
    env = gym.make(
        env_name + "-v1",
        max_episode_steps=max_episode_steps,
        device=device,
        contextual=contextual,
        answer=answer,
    )
    log_path = f"./tensorboard_log/PPO_step{n_steps}_OpenBookQA_{answer}/"
elif 'Domain' in env_name:
    answer = True
    print(env_name + "-v1")
    env = gym.make(
        env_name + "-v1",
        max_episode_steps=max_episode_steps,
        device=device,
        contextual=contextual,
        answer=answer,
    )
    log_path = f"./tensorboard_log/PPO_step{n_steps}_OpenDomain_{answer}/"
else:
    raise ValueError("env_name not found")

### load and then train model if it exists
policy = "MlpPolicy" if not args.return_image else "CnnPolicy"
model = PPO(
    policy,
    env=env,
    n_steps=n_steps,
    batch_size=50,
    verbose=1,
    gamma=0.0,
    tensorboard_log=log_path,
    stats_window_size=int(1e5),
    device=device,
)
model.learn(total_timesteps=total_timesteps, log_interval=1, progress_bar=True)
model.save(model_path)
del model  # remove to demonstrate saving and loading
print("training completed")

print(env.action_list)
