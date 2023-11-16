from typing import Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

import query.envs  # noqa: F401

bandits = {
    0: "llava-v1.5-7b",
    1: "llava-v1.5-13b",
    2: "llava-v1.5-13b-lora",
}

data_path = "synced_data/csv/waymo/"
dataset_size = 20000


class WaymoGymEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        emb_size: int = 768,
        device: str or torch.device = "cpu",
        contextual: bool = False,
        text: bool = True,
        replace_sample: bool = True,
        save_freq: int = 50,
        **kwargs,
    ) -> None:
        """
        Args:
            emb_size: size of the embedding
            device: device to run the clip model
            contextual: whether to use contextual bandit
            text: whether to use text as the observation
            replace_sample: whether to replace the sample
        """
        super(WaymoGymEnv, self).__init__()

        ### make sure the sum of p is 1
        ### Define action and observation space with discrete actions:
        n_bandits = len(bandits)
        self.action_space = spaces.Discrete(n_bandits)
        self.contextual = contextual
        self.text = text
        self.action_list = [0 for _ in range(n_bandits)]
        self.device = device
        self.emb_size = emb_size * 2 if text else emb_size
        self.replace_sample = replace_sample
        self.cnt = 0
        self.cumulative_reward = 0
        self.mean_reward_dict = {}
        self.save_freq = save_freq

        ### input is an embedding
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.emb_size,), dtype="float32"
        )

        ### load embeddings
        self.q_emb = np.load(data_path + "clip_emb_question.npy")  # 10x768
        self.img_emb = np.load(data_path + "clip_emb_img.npy")  # 2000x768
        self.arm_results = np.load(data_path + "arm_results.npy")

        ### calculate optimal reward
        opt_ = self.arm_results.max(axis=1)  # shape: (dataset_size, )
        opt_avg = opt_.mean()
        opt_ = np.cumsum(opt_) / (np.arange(opt_.shape[0]) + 1)
        assert self.arm_results.shape[1] == n_bandits, print(
            f"n_bandits should be equal to the number of {self.arm_results.shape[1]}"
        )
        print("Optimal mean reward: ", opt_avg)
        print("Best arm reward: ", self.arm_results.mean(axis=0).max())
        print("Worst arm reward: ", self.arm_results.mean(axis=0).min())
        print("arms: ", self.arm_results.mean(axis=0))

        ### shuffle the arm results
        np.random.seed(42)
        self.num_samples = self.arm_results.shape[0]
        self.shuffle_idx = np.random.choice(
            np.arange(self.num_samples), self.num_samples, replace=self.replace_sample
        )
        return

    def step(
        self, action: int, _idx: int = None
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Args:
            action: (int) the action that the agent took
            _idx: (int) the index of the next observation
        Returns:
            observation: (np.ndarray) the observation of the next step
            reward: (float) the reward from the action
            terminated: (bool) whether the episode is terminated
            truncated: (bool) whether the episode is truncated
            info: (dict) additional information
        """
        info = {}

        ### nothing is sequential here
        ### calculate reward
        current_idx = self.state
        reward = self.arm_results[current_idx, action]

        ### calculate done
        terminated = False
        truncated = False

        ### calculate next observation
        if _idx is None:
            if self.cnt >= self.num_samples:
                self.shuffle_idx = np.random.choice(
                    np.arange(self.num_samples),
                    self.num_samples,
                    replace=self.replace_sample,
                )
            next_idx = self.shuffle_idx[self.cnt % self.num_samples]
        else:
            next_idx = _idx

        ### calculate the next observation
        if self.contextual:
            if self.text:
                q_idx = next_idx % 10
                img_idx = next_idx // 10
                q_emb = self.q_emb[q_idx, :]
                img_emb = self.img_emb[img_idx]
                observation = np.concatenate((q_emb, img_emb)).astype("float32")
            else:
                img_idx = next_idx // 10
                observation = self.img_emb[img_idx].astype("float32")
        else:
            observation = np.ones((self.emb_size,), dtype="float32")

        assert observation.shape == (self.emb_size,), f"obs shape: {observation.shape}"

        ### update next state
        self.state = next_idx  # update current idx
        self.cnt += 1

        ### update action list
        self.action_list[action] += 1
        self.cumulative_reward += reward
        if self.cnt % 500 == 0:
            print(f"step: {self.cnt}, Cum Reward: ", self.cumulative_reward / self.cnt)
        if self.cnt % self.save_freq == 0 and self.cnt <= self.num_samples:
            self.mean_reward_dict[self.cnt] = self.cumulative_reward / self.cnt
        return (observation, reward, terminated, truncated, info)

    def reset(
        self,
        seed: int = None,
        _idx: int = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        """
        Args:
            seed: random seed
            _idx: (int) the index of the next observation
            _dataset: (str) the dataset to use
        Returns:
            observation: (np.ndarray) the observation of the next step
            info: (dict) additional information
        """
        super().reset(seed=seed, **kwargs)

        info = {}
        self.state = -1
        observation, _, _, _, info = self.step(0, _idx=_idx)
        return observation, info

    def save_cum_reward(self):
        ### save mean reward dict as csv
        print("Saving mean reward dict...")
        # create an empty DataFrame
        df = pd.DataFrame(columns=["Step", "mean_reward"])
        df["Step"] = self.mean_reward_dict.keys()
        df["mean_reward"] = self.mean_reward_dict.values()
        df.to_csv(
            f"synced_data/cumulative_reward/waymo_step{self.save_freq}.csv", index=False
        )
        return

    def close(self):
        self.save_cum_reward()
        return super().close()


# test emv with main function
if __name__ == "__main__":
    # Create the Gym environment
    env = WaymoGymEnv(contextual=True, text=False, replace_sample=False)
    random = True

    # Reset the environment
    obs, info = env.reset()
    cnt = 0
    terminated, truncated = False, False
    total_reward = 0

    if random:
        for i in range(20000):
            cnt += 1
            action = env.action_space.sample()
            # action = 2
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            mean_reward = total_reward / cnt
    else:
        ### test trained model
        import torch
        from stable_baselines3 import PPO

        device = "cpu"
        policy = "MlpPolicy"
        model = PPO.load(
            "synced_data/models/OpenBookQA_step100_PPO_imgFalse.zip",
            env=env,
            device=device,
        )
        idx_a_list = []
        obs, info = env.reset(_idx=0)
        for idx in range(1, 25256):
            if idx % 1000 == 0:
                print(idx)
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action, _idx=idx)
            idx_a_list.append((idx, action))

    env.close()
