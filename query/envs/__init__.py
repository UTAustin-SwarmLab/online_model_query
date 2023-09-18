### This file is used to register the environments in gym
from gymnasium.envs.registration import register

register(
    id="ImageNet1k-v1",
    max_episode_steps=1e5,
    entry_point="query.envs.ImageNet1kGymEnv:ImageNet1kGymEnv",
)

register(
    id="ImageNet1k_CIFAR100-v1",
    max_episode_steps=1e5,
    entry_point="query.envs.ImageNet1kCIFAR100GymEnv:ImageNet1kCIFAR100GymEnv",
)

register(
    id="ImageNet1k_CIFAR100Np-v1",
    max_episode_steps=1e5,
    entry_point="query.envs.ImageNet1kCIFAR100GymEnv:ImageNet1kCIFAR100GymEnvNp",
)

register(
    id="OpenBookQA-v1",
    max_episode_steps=1e5,
    entry_point="query.envs.OpenBookQAGymEnv:OpenBookQAGymEnv",
)

register(
    id="OpenDomain-v1",
    max_episode_steps=1e5,
    entry_point="query.envs.OpenDomainGymEnv:OpenDomainGymEnv",
)
