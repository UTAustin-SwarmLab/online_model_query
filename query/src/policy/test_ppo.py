### python src/policy/test_ppo.py -e ImageNet1k_CIFAR100 -d 0 -c True -n 50 -i ""
import argparse
import sys

import gymnasium as gym
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
parser.add_argument(
    "-i", "--return_image", type=bool, help="return image or not", default=False
)
parser.add_argument("-n", "--step", type=int, help="steps per update", default=100)
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
    model_path = f"./synced_data/models/{env_name}_step{args.step}_PPO_img{args.return_image}.zip"
else:
    model_path = f"./synced_data/models/{env_name}_PPO_non_contextual.zip"

set_random_seed(42, using_cuda=device != "cpu")
env = gym.make(
    env_name + "-v1",
    max_episode_steps=max_episode_steps,
    device=device,
    p=[5 / 6, 1 / 6],
    return_image=args.return_image,
    contextual=contextual,
)
### evaluate
model = PPO.load(model_path)
cumulative_reward, _, __ = env.test_sequential(
    model, "imagenet-1k"
)  # "cifar100", "imagenet-1k"
cumulative_reward, _, __ = env.test_sequential(model, "cifar100")

### save cumulative reward
# np.save(f"./synced_data/cumulative_reward/ImageNet1k_PPO_{max_episode_steps}_{total_timesteps}.npy", cumulative_reward)
