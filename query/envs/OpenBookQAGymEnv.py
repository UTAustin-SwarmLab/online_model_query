from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import query.envs  # noqa: F401

bandits = {
    # 0: "deberta-v3-base-mrqa",
    1: "deberta-v3-base-squad2",
    2: "bigbird-base-trivia-itc",
    3: "distilbert-base-uncased-distilled-squad",
    4: "roberta-base-squad2-nq",
}

datasets = {
    "SQuAD": 10507,
    "TriviaQA-web": 7785,
    "NaturalQuestionsShort": 12836,
    "NewsQA": 4212,
    "SearchQA": 16980,
    "HotpotQA": 5901,
}  # total 58221

dataset_cumsum = np.cumsum(list(datasets.values()))
print("Data_cumsum", dataset_cumsum)


class OpenBookQAGymEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        emb_size: int = 512,
        answer: bool = False,
        device: str or torch.device = "cpu",
        contextual: bool = True,
        exact_match: bool = True,
        replace_sample: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            emb_size: size of the embedding
            answer: whether to use the local model's answer as part of the observation
            device: device to run the clip model
            contextual: whether to use contextual bandit
            exact_match: whether to use exact match or f1 score
            replace_sample: whether to replace the sample
        """
        super(OpenBookQAGymEnv, self).__init__()

        ### make sure the sum of p is 1
        ### Define action and observation space with discrete actions:
        n_bandits = len(bandits)
        self.action_space = spaces.Discrete(n_bandits)
        self.contextual = contextual
        self.action_list = [0 for _ in range(n_bandits)]

        self.device = device
        self.answer = answer
        if self.answer:
            self.emb_size = emb_size * 4  # question, context*2, answer
        else:
            self.emb_size = emb_size * 3  # question, context*2

        self.local_model_name = "distilbert-base-uncased-distilled-squad"
        self.reward_range = (0, 1)
        self.cnt = 0

        ### input is an embedding
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.emb_size,), dtype="float32"
        )

        self.replace_sample = replace_sample
        ### load numpy arrays
        if exact_match:
            em = []
            for dataset in datasets.keys():
                em.append(np.load(f"./synced_data/csv/mrqa/{dataset}_exact.npy"))
            self.arm_results = np.concatenate(
                (em),
                axis=0,
            )
        else:
            f1 = []
            for dataset in datasets.keys():
                f1.append(np.load(f"./synced_data/csv/mrqa/{dataset}_f1.npy"))
            self.arm_results = np.concatenate(
                (f1),
                axis=0,
            )

        ### shuffle the arm results
        np.random.seed(42)
        self.num_samples = self.arm_results.shape[0]
        self.shuffle_idx = np.random.choice(
            np.arange(self.num_samples), self.num_samples, replace=self.replace_sample
        )
        # self.arm_results = self.arm_results[self.shuffle_idx, :]  #################
        ### remove models
        model_idx = [i for i in bandits.keys()]
        self.arm_results = self.arm_results[:, model_idx]

        assert self.arm_results.shape[1] == n_bandits, print(
            f"n_bandits should be equal to the number of {self.arm_results.shape[1]}"
        )

        ### load embeddings
        self.squad_question_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_SQuAD_question.npy"
        )
        self.squad_context_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_SQuAD_context.npy"
        )
        self.squad_answer_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_SQuAD_answer.npy"
        )
        self.squad_model_answer_np = np.load(
            f"./synced_data/csv/mrqa/clip_emb_{self.local_model_name}_SQuAD_predanswer.npy"
        )
        self.trivia_question_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_TriviaQA-web_question.npy"
        )
        self.trivia_context_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_TriviaQA-web_context.npy"
        )
        self.trivia_answer_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_TriviaQA-web_answer.npy"
        )
        self.trivia_model_answer_np = np.load(
            f"./synced_data/csv/mrqa/clip_emb_{self.local_model_name}_TriviaQA-web_predanswer.npy"
        )
        self.natural_question_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_NaturalQuestionsShort_question.npy"
        )
        self.natural_context_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_NaturalQuestionsShort_context.npy"
        )
        self.natural_answer_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_NaturalQuestionsShort_answer.npy"
        )
        self.natural_model_answer_np = np.load(
            f"./synced_data/csv/mrqa/clip_emb_{self.local_model_name}_NaturalQuestionsShort_predanswer.npy"
        )
        self.news_question_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_NewsQA_question.npy"
        )
        self.news_context_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_NewsQA_context.npy"
        )
        self.news_answer_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_NewsQA_answer.npy"
        )
        self.news_model_answer_np = np.load(
            f"./synced_data/csv/mrqa/clip_emb_{self.local_model_name}_NewsQA_predanswer.npy"
        )
        self.search_question_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_SearchQA_question.npy"
        )
        self.search_context_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_SearchQA_context.npy"
        )
        self.search_answer_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_SearchQA_answer.npy"
        )
        self.search_model_answer_np = np.load(
            f"./synced_data/csv/mrqa/clip_emb_{self.local_model_name}_SearchQA_predanswer.npy"
        )
        self.hotpot_question_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_HotpotQA_question.npy"
        )
        self.hotpot_context_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_HotpotQA_context.npy"
        )
        self.hotpot_answer_np = np.load(
            "./synced_data/csv/mrqa/clip_emb_HotpotQA_answer.npy"
        )
        self.hotpot_model_answer_np = np.load(
            f"./synced_data/csv/mrqa/clip_emb_{self.local_model_name}_HotpotQA_predanswer.npy"
        )

    def step(
        self, action: int, _idx: int = None, _dataset: str = None
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        info = {}

        ### nothing is sequential here
        ### calculate reward
        # selected_model = self.bandits[action] ### name of the model
        obs_dataset, current_idx = self.state
        reward = self.arm_results[current_idx, action]

        ### calculate done
        terminated = False
        truncated = False

        ### calculate next observation
        if _idx is None:
            next_idx = self.shuffle_idx[self.cnt]
        else:
            next_idx = _idx

        ### calculate the next dataset
        if _dataset is not None:
            next_dataset = _dataset
        else:
            for idx, dataset in enumerate(datasets.keys()):
                # print(idx, dataset, next_idx, dataset_cumsum[idx])
                if next_idx < dataset_cumsum[idx]:
                    next_dataset = dataset
                    if idx > 0:
                        ds_idx = next_idx - dataset_cumsum[idx - 1]
                    else:
                        ds_idx = next_idx
                    break

        ### calculate the next observation
        if self.contextual:
            ### load the embeddings
            if next_dataset == "SQuAD":
                question_emb = self.squad_question_np[ds_idx, :]
                context_emb = self.squad_context_np[ds_idx, :]
                model_answer_emb = self.squad_model_answer_np[ds_idx, :]
                # self.squad_answer_np[ds_idx, :]
            elif next_dataset == "TriviaQA-web":
                question_emb = self.trivia_question_np[ds_idx, :]
                context_emb = self.trivia_context_np[ds_idx, :]
                model_answer_emb = self.trivia_model_answer_np[ds_idx, :]
                # self.trivia_answer_np[ds_idx, :]
            elif next_dataset == "NaturalQuestionsShort":
                question_emb = self.natural_question_np[ds_idx, :]
                context_emb = self.natural_context_np[ds_idx, :]
                model_answer_emb = self.natural_model_answer_np[ds_idx, :]
                # self.natural_answer_np[ds_idx, :]
            elif next_dataset == "NewsQA":
                question_emb = self.news_question_np[ds_idx, :]
                context_emb = self.news_context_np[ds_idx, :]
                model_answer_emb = self.news_model_answer_np[ds_idx, :]
                # self.news_answer_np[ds_idx, :]
            elif next_dataset == "SearchQA":
                question_emb = self.search_question_np[ds_idx, :]
                context_emb = self.search_context_np[ds_idx, :]
                model_answer_emb = self.search_model_answer_np[ds_idx, :]
                # self.search_answer_np[ds_idx, :]
            elif next_dataset == "HotpotQA":
                question_emb = self.hotpot_question_np[ds_idx, :]
                context_emb = self.hotpot_context_np[ds_idx, :]
                model_answer_emb = self.hotpot_model_answer_np[ds_idx, :]
                # self.hotpot_answer_np[ds_idx, :]
            else:
                raise NotImplementedError
            if self.answer:
                # i = 0
                # for idx, dataset in enumerate(datasets.keys()):
                #     if next_dataset == dataset:
                #         i = idx
                #         break
                # observation = np.ones((self.emb_size,), dtype="float32") * i
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
            observation = np.zeros((self.emb_size,), dtype="float32")

        assert observation.shape == (self.emb_size,), f"obs shape: {observation.shape}"

        ### update next state
        self.state = next_dataset, next_idx  # update current idx
        self.cnt += 1
        if self.cnt >= self.num_samples:
            truncated = True

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
        super().reset(seed=seed, **kwargs)

        info = {}
        self.state = "_", -1
        observation, reward, terminated, truncated, info = self.step(
            0, _idx=_idx, _dataset=_dataset
        )

        return observation, info


# test emv with main function
if __name__ == "__main__":
    # Create the Gym environment
    env = OpenBookQAGymEnv(answer=True, contextual=True, exact_match=True)
    random = True
    # Reset the environment
    obs, info = env.reset()
    # Perform action loop
    cnt = 0
    terminated, truncated = False, False
    total_reward = 0

    if random:
        while not (terminated or truncated):
            cnt += 1
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if cnt % 100 == 0:
                print(cnt)
                print(obs)
                print(reward, terminated, truncated, info)
    else:
        # _ = [0, 0]
        # while not (terminated or truncated):
        #     cnt += 1
        #     obs_ = obs
        #     action = 0 if np.sum(obs) <= 200 else 1
        #     obs, reward, terminated, truncated, info = env.step(action)
        #     total_reward += reward
        #     _[action] += 1
        #     if cnt % 100 == 0:
        #         print(cnt)
        #         print(np.sum(obs_), action)
        #         print("Mean reward:", total_reward / cnt)
        #         print("Action list:", _)

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
        for idx in range(1, dataset_cumsum[-1]):
            if idx % 1000 == 0:
                print(idx)
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action, _idx=idx)
            idx_a_list.append((idx, action))

    env.close()
