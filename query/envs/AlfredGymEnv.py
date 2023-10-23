from typing import Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

import query.envs  # noqa: F401

bandits = {
    0: "Seq2Seq",
    1: "FILM",
    2: "HiTUT",
    3: "HLSM",
}


class AlfredGymEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        emb_size: int = 768,
        device: str or torch.device = "cpu",
        low_level: bool = True,
        contextual: bool = True,
        replace_sample: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            emb_size: size of the embedding
            device: device to run the clip model
            contextual: whether to use contextual bandit
            replace_sample: whether to replace the sample
        """
        super(AlfredGymEnv, self).__init__()

        ### make sure the sum of p is 1
        ### Define action and observation space with discrete actions:
        n_bandits = len(bandits)
        self.action_space = spaces.Discrete(n_bandits)
        self.contextual = contextual
        self.action_list = [0 for _ in range(n_bandits)]

        self.device = device
        self.emb_size = emb_size * 4  # question, choices

        self.replace_sample = replace_sample
        self.local_model_name = bandits[0]
        self.reward_range = (0, 1)
        self.cnt = 0

        ### input is an embedding
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.emb_size,), dtype="float32"
        )

        ### load numpy arrays
        self.arm_results = pd.load(
            "./synced_data/csv/alfred_data/alfred_merged_valid_language_goal.csv"
        )

        ### shuffle the arm results
        np.random.seed(42)
        self.num_samples = self.arm_results.shape[0]
        self.shuffle_idx = np.random.choice(
            np.arange(self.num_samples), self.num_samples, replace=self.replace_sample
        )

        ### remove models
        model_idx = [i for i in bandits.keys()]  # noqa: F401
        self.arm_results = self.arm_results.query("model in @model_idx")

        ### load embeddings
        self.instruct_np = np.load("synced_data/csv/mmlu/clip_emb_instruct.npy")
        self.ll_instruct_np = np.load(
            "synced_data/csv/mmlu/clip_emb_low_level_instruct.npy"
        )

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
        selected_row = self.arm_results.loc[
            self.arm_results["task_idx"]
            == current_idx[0] & self.arm_results["repeat_idx"]
            == current_idx[1] & self.arm_results["model"]
            == bandits[action]
        ]
        sr = selected_row["SR"]
        gc = selected_row["GC"]
        L = selected_row["L"]
        L_demo = selected_row["L*"]
        split = selected_row["split"]

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
                self.cnt = 0
            next_idx = self.shuffle_idx[self.cnt % self.num_samples]
        else:
            next_idx = _idx

        ### calculate the next observation
        if self.contextual:
            ### load the embeddings of a question and its choices and answer
            instruct_emb = self.instruct_np[next_idx, :]
            if self.low_level:
                ll_instruct_emb = self.ll_instruct_np[next_idx, :]
                observation = np.concatenate(
                    (instruct_emb, ll_instruct_emb),
                    axis=0,
                    dtype="float32",
                )
            else:
                observation = instruct_emb.astype("float32")
        else:
            observation = np.ones((self.emb_size,), dtype="float32")

        assert observation.shape == (self.emb_size,), f"obs shape: {observation.shape}"

        ### update next state
        self.state = next_idx  # update current idx
        self.cnt += 1

        ### update action list
        self.action_list[action] += 1

        return (observation, reward, terminated, truncated, info)

    def reset(
        self,
        seed: int = None,
        _idx: int = None,
        _dataset: str = None,
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
        print(f"reset: {self.cnt}")
        observation, reward, terminated, truncated, info = self.step(
            0, _idx=_idx, _dataset=_dataset
        )

        return observation, info


# test emv with main function
if __name__ == "__main__":
    # Create the Gym environment
    env = AlfredGymEnv(low_level=True, contextual=False, replace_sample=True)
    random = True

    # Reset the environment
    obs, info = env.reset()
    cnt = 0
    terminated, truncated = False, False
    total_reward = 0

    if random:
        for i in range(50000):
            cnt += 1
            action = env.action_space.sample()
            action = 5
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            cum_reward = total_reward / cnt
            if cnt % 1000 == 0:
                print(cnt)
                # print(obs)
                # print(reward, terminated, truncated, info)
                print(cum_reward)
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
