import os
import pickle
from typing import Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

import query.envs  # noqa: F401

bandits = {
    0: "FILM",
    1: "HiTUT",
    2: "HLSM",
}

data_path = "synced_data/csv/alfred_data/"
dataset_size = 13128


class AlfredGymEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        emb_size: int = 768,
        device: str or torch.device = "cpu",
        contextual: bool = True,
        low_level: bool = True,
        floor_plan: bool = True,
        replace_sample: bool = True,
        reward_metric: str = "GC",
        alpha: float = 0.05,
        beta: float = 0.005,
        **kwargs,
    ) -> None:
        """
        Args:
            emb_size: size of the embedding
            device: device to run the clip model
            contextual: whether to use contextual bandit
            low_level: whether to use low level instruction
            floor_plan: whether to use floor plan
            replace_sample: whether to replace the sample
            reward_metric: the reward metric to use
            beta: the beta parameter for the token cost
        """
        super(AlfredGymEnv, self).__init__()

        ### make sure the sum of p is 1
        ### Define action and observation space with discrete actions:
        n_bandits = len(bandits)
        self.action_space = spaces.Discrete(n_bandits)
        self.contextual = contextual
        self.action_list = [0 for _ in range(n_bandits)]

        self.device = device
        size = int(1 + 3 * low_level + 1 * floor_plan)
        self.emb_size = (
            emb_size * size
        )  # instruction + low level instruction + floorplan

        self.replace_sample = replace_sample
        self.cnt = 0
        self.cumulative_reward = 0
        self.mean_reward_dict = {}
        self.low_level = low_level
        self.floor_plan = floor_plan
        self.reward_metric = reward_metric

        ### input is an embedding
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.emb_size,), dtype="float32"
        )

        ### load embeddings
        if not (
            os.path.isfile(data_path + "clip_emb.npy")
            or os.path.isfile(data_path + "arm_results.npy")
        ):
            instruction_dict = pickle.load(
                open(data_path + "clip_emb_instruct.pkl", "rb")
            )
            low_level_instruction_dict = pickle.load(
                open(data_path + "clip_emb_low_instruct.pkl", "rb")
            )
            floorpan_dict = pickle.load(open(data_path + "floor_plan.pkl", "rb"))

            ### load csv data
            alfred_data = pd.read_csv(
                data_path + "alfred_merged_valid_language_goal.csv"
            )
            arm_results = pd.read_csv(data_path + "alfred_models_results.csv")
            emb = []
            y = np.zeros((4, len(alfred_data), len(bandits)), dtype=np.float32)
            for _, row in alfred_data.iterrows():
                task_id = row["task_idx"]
                repeat_idx = row["repeat_idx"]
                floorplan = row["task_floor"]
                emb.append(
                    np.concatenate(
                        (
                            instruction_dict[(task_id, repeat_idx)],
                            low_level_instruction_dict[(task_id, repeat_idx)],
                            floorpan_dict[floorplan],
                        )
                    )
                )
                y = np.zeros(len(bandits))
                for i, model in bandits.items():
                    result_row = arm_results.loc[
                        (arm_results["task_idx"] == task_id)
                        & (arm_results["repeat_idx"] == repeat_idx % 10)
                        & (arm_results["model"] == model)
                    ]
                    sr = result_row["SR"].iloc[0]
                    gc = result_row["GC"].iloc[0]
                    L = result_row["L"].iloc[0]
                    L_demo = result_row["L*"].iloc[0]
                    y[0, _, i] = sr
                    y[1, _, i] = gc
                    y[2, _, i] = L
                    y[3, _, i] = L_demo

            emb = np.array(emb)
            arm_results = np.array(y)
            np.save(data_path + "clip_emb.npy", emb)
            np.save(data_path + "arm_results.npy", arm_results)
        else:
            emb = np.load(data_path + "clip_emb.npy")
            arm_results = np.load(data_path + "arm_results.npy")
            print("loaded data")

        ### load token length
        token_len = np.load(data_path + "instruct_token_length.npy")  # (13128,)
        if low_level:
            token_len += np.load(data_path + "low_instruct_token_length.npy")
        token_len = token_len.reshape(dataset_size, 1)  # (13128, 1)
        token_len = np.repeat(token_len, len(bandits), axis=1)  # (39384, 3)
        token_len[:, 0] = 0

        if reward_metric == "GC":
            self.arm_results = arm_results[1, :, :]
        elif reward_metric == "PLWGC":
            self.arm_results = arm_results[1, :, :] * (
                arm_results[3, :, :]
                / np.maximum(arm_results[2, :, :], arm_results[3, :, :])
            )
        elif reward_metric == "SR":
            self.arm_results = arm_results[0, :, :]
        elif reward_metric == "PLWSR":
            self.arm_results = arm_results[0, :, :] * (
                arm_results[3, :, :]
                / np.maximum(arm_results[2, :, :], arm_results[3, :, :])
            )
        elif reward_metric == "GC+PLW":
            gc = arm_results[1, :, :]
            L_ratio = arm_results[3, :, :] / np.maximum(
                arm_results[2, :, :], arm_results[3, :, :]
            )
            L = arm_results[2, :, :]

            print(gc.shape, L_ratio.shape, token_len.shape)
            print(beta * token_len[0:5], L_ratio[:, 0:5], gc[:, 0:5], L[:, 0:5])
            # self.arm_results = 0.5 * gc + 0.5 * gc * L_ratio - beta * token_len
            self.arm_results = 0.5 * gc - alpha * np.log10(L) - beta * token_len

        ### shuffle the arm results
        np.random.seed(42)
        self.num_samples = self.arm_results.shape[0]
        self.shuffle_idx = np.random.choice(
            np.arange(self.num_samples), self.num_samples, replace=self.replace_sample
        )

        ### load embeddings
        self.instruct_np = emb[:, :emb_size]
        self.ll_instruct_np = emb[:, emb_size : emb_size * 4]
        self.floorplan_np = emb[:, emb_size * 4 :]

        opt_ = self.arm_results.max(axis=1)
        opt_avg = opt_.mean()
        opt_ = np.cumsum(opt_) / (np.arange(opt_.shape[0]) + 1)
        print("Optimal mean reward: ", opt_avg)
        print("Best arm reward: ", self.arm_results.mean(axis=0).max())
        print("Worst arm reward: ", self.arm_results.mean(axis=0).min())
        print("arms: ", self.arm_results.mean(axis=0))
        # input("Press Enter to continue...")

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
                self.cnt = 0
            next_idx = self.shuffle_idx[self.cnt % self.num_samples]
        else:
            next_idx = _idx

        ### calculate the next observation
        if self.contextual:
            ### load the embeddings of a question and its choices and answer
            obs_np = []
            instruct_emb = self.instruct_np[next_idx, :]
            obs_np.append(instruct_emb)
            if self.low_level:
                ll_instruct_emb = self.ll_instruct_np[next_idx, :]
                obs_np.append(ll_instruct_emb)
            if self.floor_plan:
                floorplan_emb = self.floorplan_np[next_idx, :]
                obs_np.append(floorplan_emb)
            observation = np.concatenate(
                obs_np,
                axis=0,
                dtype="float32",
            )
        else:
            observation = np.ones((self.emb_size,), dtype="float32")
        assert observation.shape == (self.emb_size,), f"obs shape: {observation.shape}"

        ### update next state
        self.state = next_idx  # update current idx
        self.cnt += 1
        self.cumulative_reward += reward
        if self.cnt % 1000 == 0:
            print(
                f"step: {self.cnt}, Cum Reward",
                self.cumulative_reward / self.cnt,
            )
        if self.cnt % 5 == 0:
            if self.cnt not in self.mean_reward_dict:
                self.mean_reward_dict[self.cnt] = self.cumulative_reward / self.cnt

        ### update action list
        self.action_list[action] += 1

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
        observation, _, _, _, info = self.step(
            0,
            _idx=_idx,
        )
        return observation, info

    def save_cum_reward(self):
        ### save mean reward dict as csv
        print("Saving mean reward dict...")
        # create an empty DataFrame
        df = pd.DataFrame(columns=["Step", "mean_reward"])
        df["Step"] = self.mean_reward_dict.keys()
        df["mean_reward"] = self.mean_reward_dict.values()
        df.to_csv(
            f"synced_data/cumulative_reward/alfred_{self.reward_metric}_step5.csv"
        )
        return

    def close(self):
        self.save_cum_reward()
        return super().close()


# test emv with main function
if __name__ == "__main__":
    # Create the Gym environment
    env = AlfredGymEnv(
        low_level=False,
        floor_plan=True,
        contextual=True,
        reward_metric="GC+PLW",
    )
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
            # action = 5
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            mean_reward = total_reward / cnt
            if cnt % 1000 == 0:
                print(cnt)
                # print(obs)
                # print(reward, terminated, truncated, info)
                print(mean_reward)
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
