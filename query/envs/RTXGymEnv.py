from typing import Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

import query.envs  # noqa: F401

bandits = {
    0: "small",
    1: "base",
}

data_path = "synced_data/csv/rtx/"
dataset_size = 15000


class RTXGymEnv(gym.Env):
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
        save_reward: bool = True,
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
        super(RTXGymEnv, self).__init__()

        ### Define action and observation space with discrete actions:
        n_bandits = len(bandits)
        self.action_space = spaces.Discrete(n_bandits)
        self.contextual = contextual
        self.text = text
        self.action_list = [0 for _ in range(n_bandits)]
        self.device = device
        self.emb_size = emb_size * 3 if text else emb_size * 2  # 2 for img, 1 for text
        self.replace_sample = replace_sample
        self.cnt = 0
        self.cumulative_reward = 0
        self.mean_reward_dict = {}
        self.save_freq = save_freq
        self.save_reward = save_reward

        ### input is an embedding
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.emb_size,), dtype="float32"
        )

        ### load embeddings
        bridge = np.load(data_path + "bridge_instruct_emb.npy")
        kuka = np.load(data_path + "kuka_instruct_emb.npy")
        fractal = np.load(data_path + "fractal20220817_data_instruct_emb.npy")
        self.q_emb = np.concatenate((bridge, kuka, fractal), axis=0)  # 15000 x 768
        assert self.q_emb.shape[0] == dataset_size, print(
            f"q_emb shape: {self.q_emb.shape}"
        )

        bridge = np.load(data_path + "bridge_img_emb.npy")
        kuka = np.load(data_path + "kuka_img_emb.npy")
        fractal = np.load(data_path + "fractal20220817_data_img_emb.npy")
        self.img_emb = np.concatenate((bridge, kuka, fractal), axis=0)  # 15000 x 1536
        assert self.q_emb.shape[0] == self.img_emb.shape[0], print(
            f"q_emb shape: {self.q_emb.shape}, img_emb shape: {self.img_emb.shape}"
        )

        small_bridge = np.load("synced_data/rtx/bridge_small_action_errors.npy")
        small_kuka = np.load("synced_data/rtx/kuka_small_action_errors.npy")
        small_fractal = np.load(
            "synced_data/rtx/fractal20220817_data_small_action_errors.npy"
        )
        base_bridge = np.load("synced_data/rtx/bridge_base_action_errors.npy")
        base_kuka = np.load("synced_data/rtx/kuka_base_action_errors.npy")
        base_fractal = np.load(
            "synced_data/rtx/fractal20220817_data_base_action_errors.npy"
        )
        base = np.concatenate((base_bridge, base_kuka, base_fractal), axis=0)
        small = np.concatenate((small_bridge, small_kuka, small_fractal), axis=0)
        self.arm_results = np.stack((small, base), axis=1)  # 1500 x 2
        # make it negative (cost to reward) then scale it
        self.arm_results *= -20
        assert self.arm_results.shape == (dataset_size, n_bandits), print(
            f"arm_results shape: {self.arm_results.shape}"
        )

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
        self.state = 202
        # print(self.arm_results[self.state, 1])
        # ## print rows with values between
        # print(self.arm_results[(self.arm_results > -0.9) & (self.arm_results < -0.7)])
        # ## print index of rows with values between -1.5 and -0.5
        # print(np.where((self.arm_results > -0.9) & (self.arm_results < -0.7)))
        # input()
        observation, r, _, _, info = self.step(1, _idx=_idx)
        return observation, info

    def save_cum_reward(self):
        ### save mean reward dict as csv
        if self.save_reward:
            print("Saving mean reward dict...")
            # create an empty DataFrame
            df = pd.DataFrame(columns=["Step", "mean_reward"])
            df["Step"] = self.mean_reward_dict.keys()
            df["mean_reward"] = self.mean_reward_dict.values()
            df.to_csv(
                f"synced_data/cumulative_reward/rtx_step{self.save_freq}.csv",
                index=False,
            )
        return

    def close(self):
        if self.save_reward:
            self.save_cum_reward()
        return super().close()


# test emv with main function
if __name__ == "__main__":
    # Create the Gym environment
    env = RTXGymEnv(contextual=True, text=False, replace_sample=False)
    random = True

    # Reset the environment
    obs, info = env.reset()
    cnt = 0
    terminated, truncated = False, False
    total_reward = 0

    if random:
        for i in range(dataset_size):
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
