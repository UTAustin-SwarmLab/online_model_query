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
        alpha: float = 0.03,
        beta: float = 0.0008,
        save_freq: int = 50,
        save_reward: bool = True,
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
            beta: the weight of token costs. For OpenAI, the cost is $0.02/1000 tokens
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
        if self.answer:
            self.emb_size = emb_size * 3  # question, choices, answer
        else:
            self.emb_size = emb_size * 2  # question, choices

        self.replace_sample = replace_sample
        self.reward_range = (0, 1)
        self.cnt = 0
        self.cumulative_reward = 0
        self.mean_reward_dict = {}
        self.save_freq = save_freq
        self.save_reward = save_reward
        self.acc_list = []
        self.cost_list = []
        self.alpha = alpha
        self.beta = beta

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
        self.arm_results = self.arm_results[selected_indices, :]
        self.arm_results = self.arm_results[:, model_idx]

        assert self.arm_results.shape[1] == n_bandits, print(
            f"n_bandits should be equal to the number of {self.arm_results.shape[1]}"
        )

        ### load latency data
        self.model_latency = np.zeros_like(self.arm_results)
        idx = 0
        for key, value in bandits.items():
            latency = pd.read_csv(f"synced_data/csv/mmlu/{value}_answer_time.csv")
            latency = np.array(latency["answer_time"] + latency["load_time"])
            repeat_cnt = self.arm_results.shape[0] // len(latency) + 1
            latency = np.tile(latency, repeat_cnt)
            self.model_latency[:, idx] = latency[selected_indices]
            idx += 1

        ### load token costs
        self.token_len = np.zeros_like(self.arm_results)
        token_length = np.load("synced_data/csv/mmlu/question_token_length.npy")
        self.token_len[:, 1:] = token_length[selected_indices, np.newaxis]

        ### add acc and latency
        self.reward = (
            self.arm_results
            - alpha * np.log10(self.model_latency)
            - beta * self.token_len
        )
        print(
            f"reward = {self.arm_results[-1]} - {alpha * np.log10(self.model_latency[-1])} - {beta*self.token_len[-1]}"
        )

        ### calculate optimal reward
        opt_ = self.reward.max(axis=1)  # shape: (dataset_size, )
        opt_avg = opt_.mean()
        opt_ = np.cumsum(opt_) / (np.arange(opt_.shape[0]) + 1)
        print("Optimal mean reward: ", opt_avg)
        print("Best arm reward: ", self.reward.mean(axis=0).max())
        print("Worst arm reward: ", self.reward.mean(axis=0).min())
        print("arms: ", self.reward.mean(axis=0))

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
        return

    def step(
        self,
        action: int,
        _idx: int = None,
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
        reward = self.reward[current_idx, action]
        self.acc_list.append(self.arm_results[current_idx, action])
        self.cost = (
            -self.alpha * np.log10(self.model_latency[current_idx, action])
            - self.beta * self.token_len[current_idx, action]
        )
        self.cost_list.append(self.cost)

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

        ### update next state
        self.state = next_idx  # update current idx
        self.cnt += 1

        ### update action list
        self.action_list[action] += 1
        self.cumulative_reward += reward
        if self.cnt % 500 == 0:
            print(f"step: {self.cnt}, Cum Reward:", self.cumulative_reward / self.cnt)
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
        observation, _, _, _, info = self.step(1, _idx=_idx)
        return observation, info

    def save_cum_reward(self):
        ### save mean reward dict as csv
        print("Saving mean reward dict...")
        # create an empty DataFrame
        df = pd.DataFrame(columns=["Step", "mean_reward"])
        df["Step"] = self.mean_reward_dict.keys()
        df["mean_reward"] = self.mean_reward_dict.values()
        df.to_csv(
            f"synced_data/cumulative_reward/mmlu_latency_step{self.save_freq}.csv",
            index=False,
        )
        return

    def close(self):
        if self.save_reward:
            self.save_cum_reward()
        print("Mean of acc list: ", sum(self.acc_list) / len(self.acc_list))
        print("Mean of latency list: ", sum(self.cost_list) / len(self.cost_list))
        print("len list: ", len(self.acc_list))
        return super().close()


# test emv with main function
if __name__ == "__main__":
    # Create the Gym environment
    env = OpenDomainLatencyGymEnv(
        answer=True, contextual=False, exact_match=True, save_reward=False
    )
    random = True

    # Reset the environment
    obs, info = env.reset()
    cnt = 0
    terminated, truncated = False, False
    total_reward = 0

    if random:
        for i in range(10000):
            cnt += 1
            action = env.action_space.sample()
            action = 0  # Mean of acc list:  0.5986401359864013 Mean of latency list:  -0.0421425550265716
            action = 1  # Mean of acc list:  0.76002399760024 Mean of latency list: -0.20102083384740577
            action = 2  # Mean of acc list:  0.7706229377062294 Mean of latency list:  -0.17280960432022513
            action = 3  # Mean of acc list: 0.6708329167083291 Mean of latency list:  -0.1501285699839447
            obs, reward, terminated, truncated, info = env.step(action)

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

    print(env.action_list)
    env.close()
