from typing import Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

import query.envs  # noqa: F401

bandits = {
    # 0: "vicuna-7b-v1.5",
    # 1: "falcon-180B",
    2: "falcon-180B-chat",
    # 3: "qCammel-70-x",
    4: "Llama-2-70b-instruct",
    # 5: "Llama-2-70b-instruct-v2",
    # 6: "StableBeluga-13B",
    7: "airoboros-l2-70b",
}

# subdatasets = json.load(open("synced_data/mmlu/subdatasets.json"))
subsets_dict = {
    "0": "arc:challenge",
    "1": "hellaswag",
    "2": "hendrycksTest-abstract_algebra",
    "3": "hendrycksTest-anatomy",
    "4": "hendrycksTest-astronomy",
    "5": "hendrycksTest-business_ethics",
    "6": "hendrycksTest-clinical_knowledge",
    "7": "hendrycksTest-college_biology",
    "8": "hendrycksTest-college_chemistry",
    "9": "hendrycksTest-college_computer_science",
    "10": "hendrycksTest-college_mathematics",
    "11": "hendrycksTest-college_medicine",
    "12": "hendrycksTest-college_physics",
    "13": "hendrycksTest-computer_security",
    "14": "hendrycksTest-conceptual_physics",
    "15": "hendrycksTest-econometrics",
    "16": "hendrycksTest-electrical_engineering",
    "17": "hendrycksTest-elementary_mathematics",
    "18": "hendrycksTest-formal_logic",
    "19": "hendrycksTest-global_facts",
    "20": "hendrycksTest-high_school_biology",
    "21": "hendrycksTest-high_school_chemistry",
    "22": "hendrycksTest-high_school_computer_science",
    "23": "hendrycksTest-high_school_european_history",
    "24": "hendrycksTest-high_school_geography",
    "25": "hendrycksTest-high_school_government_and_politics",
    "26": "hendrycksTest-high_school_macroeconomics",
    "27": "hendrycksTest-high_school_mathematics",
    "28": "hendrycksTest-high_school_microeconomics",
    "29": "hendrycksTest-high_school_physics",
    "30": "hendrycksTest-high_school_psychology",
    "31": "hendrycksTest-high_school_statistics",
    "32": "hendrycksTest-high_school_us_history",
    "33": "hendrycksTest-high_school_world_history",
    "34": "hendrycksTest-human_aging",
    "35": "hendrycksTest-human_sexuality",
    "36": "hendrycksTest-international_law",
    "37": "hendrycksTest-jurisprudence",
    "38": "hendrycksTest-logical_fallacies",
    "39": "hendrycksTest-machine_learning",
    "40": "hendrycksTest-management",
    "41": "hendrycksTest-marketing",
    "42": "hendrycksTest-medical_genetics",
    "43": "hendrycksTest-miscellaneous",
    "44": "hendrycksTest-moral_disputes",
    "45": "hendrycksTest-moral_scenarios",
    "46": "hendrycksTest-nutrition",
    "47": "hendrycksTest-philosophy",
    "48": "hendrycksTest-prehistory",
    "49": "hendrycksTest-professional_accounting",
    "50": "hendrycksTest-professional_law",
    "51": "hendrycksTest-professional_medicine",
    "52": "hendrycksTest-professional_psychology",
    "53": "hendrycksTest-public_relations",
    "54": "hendrycksTest-security_studies",
    "55": "hendrycksTest-sociology",
    "56": "hendrycksTest-us_foreign_policy",
    "57": "hendrycksTest-virology",
    "58": "hendrycksTest-world_religions",
}


class OpenDomainGymEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        emb_size: int = 768,
        answer: bool = False,
        device: str or torch.device = "cpu",
        contextual: bool = True,
        replace_sample: bool = False,
        max_steps: int = 1000,
        **kwargs,
    ) -> None:
        """
        Args:
            emb_size: size of the embedding
            answer: whether to use the local model's answer as part of the observation
            device: device to run the clip model
            contextual: whether to use contextual bandit
            replace_sample: whether to replace the sample
        """
        super(OpenDomainGymEnv, self).__init__()

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
        # self.local_model_name = bandits[0]
        self.reward_range = (0, 1)
        self.cnt = 0
        self.cumulative_reward = 0
        self.max_steps = max_steps

        ### input is an embedding
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.emb_size,), dtype="float32"
        )

        ### load subsets ###
        subsets = pd.read_csv("synced_data/csv/mmlu/vicuna-7b-v1.5_nochoice.csv")
        subset_map = subsets_dict
        selected_indices = []
        idx = 0
        for _, row in subsets.iterrows():
            if row["subdataset"] in subset_map.values():
                selected_indices.append(idx)
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
        print(
            "Embeddings loaded. Shape: ",
            self.question_np.shape,
            self.context_np.shape,
            self.model_answer_np.shape,
            self.arm_results.shape,
        )
        # input()

    def step(
        self,
        action: int,
        _idx: int = None,
        _dataset: str = None,
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
            for key, value in self.subset_map.items():
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
            0, _idx=_idx, _dataset=_dataset
        )
        return observation, info


# test emv with main function
if __name__ == "__main__":
    # Create the Gym environment
    env = OpenDomainGymEnv(answer=True, contextual=False, exact_match=True)
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
