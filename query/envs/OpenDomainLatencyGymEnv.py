import json
from typing import Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

import query.envs  # noqa: F401

bandits = {
    0: "vicuna-7b-v1.5",
    # 1: "falcon-180B",
    2: "falcon-180B-chat",
    # 3: "qCammel-70-x",
    4: "Llama-2-70b-instruct",
    # 5: "Llama-2-70b-instruct-v2",
    6: "StableBeluga-13B",
    # 7: "airoboros-l2-70b",
}

subset_map = json.load(open("synced_data/mmlu/subdatasets.json"))


class OpenDomainLatencyGymEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        emb_size: int = 768,
        answer: bool = False,
        device: str or torch.device = "cpu",
        contextual: bool = True,
        replace_sample: bool = False,
        alpha: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Args:
            emb_size: size of the embedding
            answer: whether to use the local model's answer as part of the observation
            device: device to run the clip model
            contextual: whether to use contextual bandit
            replace_sample: whether to replace the sample
            alpha: the weight of latency
        """
        super(OpenDomainLatencyGymEnv, self).__init__()

        ### make sure the sum of p is 1
        ### Define action and observation space with discrete actions:
        n_bandits = len(bandits)
        self.action_space = spaces.Discrete(n_bandits)
        self.contextual = contextual
        self.action_list = [0 for _ in range(n_bandits)]

        self.device = device
        self.answer = answer
        self.alpha = alpha
        if self.answer:
            self.emb_size = emb_size * 3  # question, choices, answer
        else:
            self.emb_size = emb_size * 2  # question, choices

        self.replace_sample = replace_sample
        # self.local_model_name = bandits[0]
        self.reward_range = (0, 1)
        self.cnt = 0
        self.nstep = 0
        self.cumulative_reward = 0

        ### input is an embedding
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.emb_size,), dtype="float32"
        )

        ### load subsets ###
        subsets = pd.read_csv("synced_data/csv/mmlu/vicuna-7b-v1.5_nochoice.csv")
        selected_indices = []
        self.subsets = []
        idx = 0
        for _, row in subsets.iterrows():
            if row["subdataset"] in subset_map.values():
                selected_indices.append(idx)
                self.subsets.append(row["subdataset"])
            idx += 1
        print(f"selected indices: {len(selected_indices)}")

        ### shuffle the arm results
        np.random.seed(42)
        self.num_samples = len(selected_indices)
        self.shuffle_idx = np.random.choice(
            np.arange(self.num_samples), self.num_samples, replace=self.replace_sample
        )

        ### remove models
        model_idx = [i for i in bandits.keys()]
        self.arm_results = np.load(
            "synced_data/csv/mmlu/models_accnorm.npy"
        )  # shape = [25256, 8]
        print(
            f"Arm results shape: {self.arm_results.shape}",
            model_idx,
        )
        self.arm_results = self.arm_results[selected_indices, :]
        self.arm_results = self.arm_results[:, model_idx]

        assert self.arm_results.shape[1] == n_bandits, print(
            f"n_bandits should be equal to the number of {self.arm_results.shape[1]}"
        )

        ### load embeddings
        self.question_np = np.load("synced_data/csv/mmlu/clip_emb_question.npy")[
            selected_indices, :
        ]
        self.context_np = np.load("synced_data/csv/mmlu/clip_emb_choices.npy")[
            selected_indices, :
        ]
        self.model_answer_np = (np.load("synced_data/csv/mmlu/clip_emb_answer.npy"))[
            selected_indices, :
        ]

        ### load latency data
        self.model_latency_list = []
        self.len_model_latency = []
        for key, value in bandits.items():
            latency = pd.read_csv(f"synced_data/csv/mmlu/{value}_answer_time.csv")
            latency = np.array(latency["answer_time"] + latency["load_time"])
            print(f"Loaded: {key}, {latency.shape}")
            self.model_latency_list.append(latency)
            self.len_model_latency.append(len(latency))

        assert n_bandits == len(self.model_latency_list), print(
            f"n_bandits should be equal to the number of {len(self.model_latency_list)}"
        )
        return

    def step(
        self,
        action: int,
        _idx: int = None,
        _dataset: str = None,
        reset: bool = False,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Args:
            action: (int) the action that the agent took
            _idx: (int) the index of the next observation
            _dataset: (str) the dataset to use
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
        ### add latency
        latency = self.model_latency_list[action][
            current_idx % self.len_model_latency[action]
        ]
        reward = self.arm_results[current_idx, action] - self.alpha * latency
        print(
            f"reward: {reward}, acc: {self.arm_results[current_idx, action]}, latency: {latency}"
        )

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
            ### load the embeddings of a question and its choices and answer
            question_emb = self.question_np[next_idx, :]
            context_emb = self.context_np[next_idx, :]
            if self.answer:
                model_answer_emb = self.model_answer_np[next_idx, :]
                observation = np.concatenate(
                    (question_emb, context_emb, model_answer_emb),
                    axis=0,
                    dtype="float32",
                )
            else:
                observation = np.concatenate(
                    (question_emb, context_emb), axis=0, dtype="float32"
                )
        else:
            subset = self.subsets[next_idx]
            subset_idx = 0
            for key, value in subset_map.items():
                if subset == value:
                    subset_idx = int(key)
                    break
            observation = np.ones((self.emb_size,), dtype="float32") * subset_idx

        assert observation.shape == (self.emb_size,), f"obs shape: {observation.shape}"

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
        # if self.cnt % 500 == 0:
        #     print(f"reset: {self.cnt}")
        observation, reward, terminated, truncated, info = self.step(
            0, _idx=_idx, _dataset=_dataset, reset=True
        )
        return observation, info

    def close(self):
        ### save mean reward dict as csv
        print("Saving mean reward dict...")
        # create an empty DataFrame
        df = pd.DataFrame(columns=["Step", "mean_reward"])
        df["Step"] = self.mean_reward_dict.keys()
        df["mean_reward"] = self.mean_reward_dict.values()
        df.to_csv("synced_data/cumulative_reward/mmlu_latency_step5.csv", index=False)
        return super().close()


# test emv with main function
if __name__ == "__main__":
    # Create the Gym environment
    env = OpenDomainLatencyGymEnv(answer=True, contextual=False, exact_match=True)
    random = True

    # Reset the environment
    obs, info = env.reset()
    cnt = 0
    terminated, truncated = False, False
    total_reward = 0

    if random:
        for i in range(5000):
            cnt += 1
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            mean_reward = total_reward / cnt
            if cnt % 500 == 0:
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
