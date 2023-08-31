### tensorboard --logdir='./tensorboard_log/PPO_ImageNet1k' --port=6006
### python src/record_embed.py -e ImageNet1k_CIFAR100 -d 0 -c True
import argparse
import sys

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

sys.path.append("./")


parser = argparse.ArgumentParser(description="Create Configuration")
parser.add_argument("-e", "--env", type=str, help="environment name", default="")
parser.add_argument("-d", "--device", type=int, help="device name", default=-1)
parser.add_argument(
    "-c", "--contextual", type=bool, help="comtxtual of not", default=False
)
args = parser.parse_args()
print(args)

env_name = args.env  # 'ImageNet1k', 'ImageNet1k_CIFAR100'
contextual = args.contextual  # False
device = (
    f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
)
max_episode_steps = 6000
total_timesteps = max_episode_steps * 100

if contextual:
    model_path = f"./synced_data/models/{env_name}_PPO.zip"
else:
    model_path = f"./synced_data/models/{env_name}_PPO_non_contextual.zip"

set_random_seed(42, using_cuda=device != "cpu")
env = gym.make(
    env_name + "-v1",
    max_episode_steps=max_episode_steps,
    device=device,
    p=[0.5, 0.5],
    return_image=False,
    contextual=contextual,
)

### evaluate
model = PPO.load(model_path)
_, obs_list1, __ = env.test_sequential(
    model, "imagenet-1k"
)  # "cifar100", "imagenet-1k"
# _, obs_list2, __ = env.test_sequential(model, "cifar100")

### save obs_list
obs_list1 = np.array(obs_list1)
print(obs_list1.shape)
np.save("./synced_data/csv/clip_emb_imagenet-1k.npy", obs_list1)

# obs_list2 = np.array(obs_list2)
# print(obs_list2.shape)
# np.save(f"./synced_data/csv/clip_emb_cifar100.npy", obs_list2)
