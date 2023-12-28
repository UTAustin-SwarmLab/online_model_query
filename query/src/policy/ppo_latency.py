### poetry run python query/src/policy/ppo_latency.py -e OpenDomain -c True -n 5
### poetry run python query/src/policy/ppo_latency.py -e Alfred -c True -n 5
### poetry run python query/src/policy/ppo_latency.py -e Waymo -c True -n 5
import argparse

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

import query.envs  # noqa: F401

parser = argparse.ArgumentParser(description="Create Configuration")
parser.add_argument("-e", "--env", type=str, help="environment name", default="")
parser.add_argument("-d", "--device", type=int, help="device name", default=-1)
parser.add_argument(
    "-c", "--contextual", type=bool, help="comtxtual of not", default=False
)
parser.add_argument(
    "-i", "--return_image", type=bool, help="return image or not", default=False
)
parser.add_argument("-n", "--step", type=int, help="steps per update", default=25)
args = parser.parse_args()
print(args)

# 'ImageNet1k', 'ImageNet1k_CIFAR100', 'ImageNet1k_CIFAR100Np', 'OpenBookQA', 'Alfred-v1'
env_name = args.env
contextual = args.contextual  # False
device = (
    f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
)
max_episode_steps = args.step
n_steps = args.step

if contextual:
    model_path = (
        f"./synced_data/models/{env_name}_step{n_steps}_PPO_img{args.return_image}.zip"
    )
else:
    model_path = f"./synced_data/models/{env_name}_step{n_steps}_PPO_non_contextual_img{args.return_image}.zip"

set_random_seed(42, using_cuda=device != "cpu")
if "Domain" in env_name:
    total_timesteps = 10000
    answer = False
    # answer = True
    print(env_name + "-v1")
    env = gym.make(
        "OpenDomainLatency-v1",
        max_episode_steps=max_episode_steps,
        device=device,
        contextual=contextual,
        answer=answer,
        replace_sample=False,
        max_steps=n_steps,
        save_freq=n_steps,
    )
    tag = ""
    if contextual:
        tag += "_contextual"
    if answer:
        tag += "_answer"
    log_path = f"./tensorboard_log/PPO_latency_step{n_steps}_OpenDomain{tag}/"
elif "Waymo" in env_name:
    total_timesteps = 20000
    env = gym.make(
        "WaymoLatency-v1",
        max_episode_steps=max_episode_steps,
        device=device,
        contextual=contextual,
        text=True,
        replace=False,
        save_freq=n_steps,
    )
    log_path = f"./tensorboard_log/PPO_latency_step{n_steps}_Waymo/"
else:
    raise ValueError("env_name not found: " + env_name)

### train agent model
### hyperparameters of PPO to tune:
policy = "MlpPolicy" if not args.return_image else "CnnPolicy"
policy_kwargs = dict(
    activation_fn=torch.nn.GELU,
    # net_arch=dict(pi=[768 * 2, 512, 256], vf=[768 * 2, 512, 256]),
    net_arch=[768, 512, 128, 64, 16],
)
model = PPO(
    policy,
    learning_rate=1e-4,  # 3e-4
    target_kl=0.003,  # (0.003 to 0.03)
    ent_coef=0.01,
    policy_kwargs=policy_kwargs,
    env=env,
    n_steps=n_steps,
    batch_size=n_steps,
    verbose=0,
    gamma=0.0,
    # tensorboard_log=log_path,
    stats_window_size=int(1),
    device=device,
)
model.learn(total_timesteps=total_timesteps, log_interval=1, progress_bar=True)
model.save(model_path)
del model  # remove to demonstrate saving and loading
print("training completed")

print(env.action_list)
env.close()
