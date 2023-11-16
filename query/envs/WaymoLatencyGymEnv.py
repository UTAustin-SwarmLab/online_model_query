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


class WaymoLatencyGymEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        emb_size: int = 768,
        device: str or torch.device = "cpu",
        contextual: bool = False,
        text: bool = True,
        replace_sample: bool = True,
        alpha: float = 0.2,
        beta: float = 0.01 / 10,
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
            alpha: the weight of latency
            beta: the weight of token costs. For OpenAI, the cost is $0.02/1000 tokens

            Note:
            A 1920x1080 image with 3-byte pixels is approximately 6.22 megabytes (MB).
            This is because each pixel uses 3 bytes.
            5G home internet commonly gives you speeds around 100~300 Mbps.
            Thus, the transmission time is 6.22MB * 8 / 300Mbps = 0.166s
        """
        super(WaymoLatencyGymEnv, self).__init__()

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
        self.q_emb = np.tile(self.q_emb, (2000, 1))  # 20000x768
        self.token_len = np.load(data_path + "question_token_length.npy")  # 10
        self.token_len = np.tile(self.token_len, 2000)
        self.token_len = self.token_len.reshape(dataset_size, 1)  # 20000x1
        self.token_len = np.repeat(self.token_len, 3, axis=1)  # 20000x3
        self.token_len[:, 0] = 0  # no need to pay for the token cost
        self.img_emb = np.load(data_path + "clip_emb_img.npy")  # 2000x768
        self.img_emb = np.repeat(self.img_emb, 10, axis=0)
        self.arm_results = np.load(data_path + "arm_results.npy")  # 20000x3
        self.model_latency = np.load(data_path + "arm_results_time.npy")  # 20000x3
        assert self.q_emb.shape == self.img_emb.shape, print(
            f"q and img shape: {self.q_emb.shape} != {self.img_emb.shape}"
        )
        assert self.token_len.shape == self.arm_results.shape, print(
            f"token_len shape 0: {self.token_len.shape} should be {self.arm_results.shape}"
        )
        self.img_mean = np.mean(self.img_emb, axis=0)
        self.img_std = np.std(self.img_emb, axis=0)
        self.q_mean = np.mean(self.q_emb, axis=0)
        self.q_std = np.std(self.q_emb, axis=0)

        ### add image transmission time
        self.model_latency[:, 1:] += 0.166 * 2

        ### add acc and latency
        self.reward = (
            self.arm_results
            - alpha * np.log10(self.model_latency)
            - beta * self.token_len * 2
        )
        print(
            f"reward = {self.arm_results[-1]} - {alpha*np.log10(self.model_latency[-1])} - {beta*self.token_len[-1]*2}"
        )
        print("Model latency average: ", np.mean(self.model_latency, axis=0))
        print("Model acc average: ", np.mean(self.arm_results, axis=0))
        print("Model token cost average: ", np.mean(self.token_len, axis=0))

        ### calculate optimal reward
        opt_ = self.reward.max(axis=1)  # shape: (dataset_size, )
        opt_avg = opt_.mean()
        opt_ = np.cumsum(opt_) / (np.arange(opt_.shape[0]) + 1)
        print("Optimal mean reward: ", opt_avg)
        print("Best arm reward: ", self.reward.mean(axis=0).max())
        print("Worst arm reward: ", self.reward.mean(axis=0).min())
        print("arms: ", self.reward.mean(axis=0))

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
        reward = self.reward[current_idx, action]

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
                q_emb = (self.q_emb[next_idx, :] - self.q_mean) / self.q_std
                img_emb = (self.img_emb[next_idx, :] - self.img_mean) / self.img_std
                observation = np.concatenate((q_emb, img_emb)).astype("float32")
            else:
                next_idx
                observation = self.img_emb[next_idx].astype("float32")
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
            print(f"step: {self.cnt}, Cum Reward", self.cumulative_reward / self.cnt)
            print("action list: ", self.action_list)
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
            f"synced_data/cumulative_reward/waymo_latency_step{self.save_freq}.csv",
            index=False,
        )
        return

    def close(self):
        self.save_cum_reward()
        return super().close()


# test emv with main function
if __name__ == "__main__":
    # Create the Gym environment
    env = WaymoLatencyGymEnv(contextual=True, text=False, replace_sample=False)
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
            action = 0
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
