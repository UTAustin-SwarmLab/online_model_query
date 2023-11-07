from typing import Tuple

import gymnasium as gym
import numpy as np
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
        super(WaymoGymEnv, self).__init__()

        ### make sure the sum of p is 1
        ### Define action and observation space with discrete actions:
        n_bandits = len(bandits)
        self.action_space = spaces.Discrete(n_bandits)
        self.contextual = contextual
        self.action_list = [0 for _ in range(n_bandits)]

        self.device = device
        self.emb_size = emb_size  # instruction + low level instruction + floorplan

        self.replace_sample = replace_sample
        self.cnt = 0
        self.cumulative_reward = 0

        ### input is an embedding
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.emb_size,), dtype="float32"
        )

        ### load embeddings
        emb = np.load(data_path + "clip_emb.npy")
        arm_results = np.load(data_path + "arm_results.npy")
        print("loaded data")

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
            plw = arm_results[1, :, :] * (
                arm_results[3, :, :]
                / np.maximum(arm_results[2, :, :], arm_results[3, :, :])
            )
            self.arm_results = 0.5 * gc + 0.5 * plw

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

        ### update action list
        self.action_list[action] += 1
        self.cumulative_reward += reward
        if self.cnt % 1000 == 0:
            print(f"step: {self.cnt}, Cum Reward", self.cumulative_reward / self.cnt)

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
        print(f"reset: {self.cnt}")
        observation, _, _, _, info = self.step(
            0,
            _idx=_idx,
        )
        self.cnt -= 1  ### Bug...
        return observation, info


# test emv with main function
if __name__ == "__main__":
    # Create the Gym environment
    env = WaymoGymEnv(low_level=True, floor_plan=True, contextual=True)
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
