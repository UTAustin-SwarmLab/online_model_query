import gymnasium as gym
import omegaconf
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import query.envs  # noqa: F401
import wandb
from wandb.integration.sb3 import WandbCallback

cfg = omegaconf.OmegaConf.load("query/src/policy/configs/ppo.yaml")


def run_experiment(cfg: omegaconf.DictConfig):
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 500,
        "env_name": "WaymoLatency-v1",
    }
    dict_cfg = omegaconf.OmegaConf.to_container(cfg)
    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    def make_env():
        env = gym.make(config["env_name"])
        env = Monitor(env)  # record stats such as returns
        return env

    env = DummyVecEnv([make_env])
    model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=10,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    y = list(env.get_attr("mean_reward_dict")[0].values())
    print(y, type(y))

    metrics = {}
    for idx, yy in enumerate(y):
        metrics["mean_reward"] = yy
        run.log(metrics)
    run.finish()


if __name__ == "__main__":
    run_experiment(cfg)
