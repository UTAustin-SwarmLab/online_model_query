import random

import clip
import gymnasium as gym
import numpy as np
import pandas as pd
from datasets import load_dataset
from gymnasium import spaces
from PIL import Image

val_set_size = 50000


class ImageNet1kGymEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_bandits=5,
        h=512,
        w=512,
        c=3,
        emb_size=512,
        device="cpu",
        return_image=False,
        contextual=True,
        **kwargs,
    ):
        super(ImageNet1kGymEnv, self).__init__()
        ### Define action and observation space with discrete actions:
        self.action_space = spaces.Discrete(n_bandits)
        self.bandits = {
            0: "convnext",
            1: "mit",
            2: "mobilenet",
            3: "resnet",
            4: "ViT",
        }
        self.contextual = contextual
        self.model_result_df = {}
        assert len(self.bandits) == n_bandits, print(
            "n_bandits should be equal to the number of bandits"
        )

        self.return_image = return_image
        self.emb_size = emb_size
        self.device = device
        self.reward_range = (0, 1)
        for idx, model in self.bandits.items():
            csv_path = f"./synced_data/csv/imagenet/{model}_df_validation.csv"
            self.model_result_df[idx] = pd.read_csv(csv_path)

        ### input is an image or an embedding
        if return_image:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(h, w, c), dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(emb_size,), dtype="float32"
            )
            self.model, self.preprocess = clip.load(
                "ViT-B/32", jit=False, device=self.device
            )

        self.dataset = load_dataset(
            "imagenet-1k", split="validation", streaming=False, use_auth_token=True
        ).with_format("numpy")

    def step(self, action: int, _idx: int = None):
        info = {}
        ### nothing is sequential here
        ### calculate reward
        df_validation = self.model_result_df[action]  ### df of the model
        current_idx = self.state
        correctTF = df_validation.iloc[current_idx]["correctTF"]
        reward = 1 if correctTF else 0

        ### calculate done
        self.cnt += 1
        # terminated = self.cnt >= self.max_steps
        terminated = False
        truncated = False

        ### force to select the next observation
        if _idx is not None:
            idx = _idx
        else:
            ### calculate next observation
            idx = random.randint(0, val_set_size - 1)
        np_image, label = self.dataset[idx]["image"], self.dataset[idx]["label"].item()

        if self.contextual:
            ### convert grayscale to RGB
            if len(np_image.shape) == 2:
                np_image = np.stack((np_image,) * 3, axis=-1)
            if self.return_image:
                observation = np_image
            else:
                image = (
                    self.preprocess(Image.fromarray(np_image, "RGB"))
                    .unsqueeze(0)
                    .to(self.device)
                )
                image_features = self.model.encode_image(image)
                observation = image_features.reshape((-1)).detach().cpu().numpy()
                assert observation.shape == (self.emb_size,), print(
                    "obs shape: ", observation.shape
                )
        else:
            observation = np.zeros((self.emb_size,), dtype="float32")

        ### update next state
        self.state = idx  # update current idx

        # if self.cnt % 1000 == 0:
        # print("cnt: ", self.cnt, "current state: ", current_idx, "action: ", action, "next state: ", idx)
        # print("new state: ", self.state, "reward: ", reward, "terminated: ", terminated, "truncated: ", truncated)

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed=None,
        _idx=None,
        **kwargs,
    ):
        super().reset(seed=seed, **kwargs)
        info = {}
        ### calculate the 1st observation
        if _idx is not None:
            idx = _idx
        else:
            idx = random.randint(0, val_set_size)
        np_image, label = self.dataset[idx]["image"], self.dataset[idx]["label"].item()

        if self.contextual:
            ### convert grayscale to RGB
            if len(np_image.shape) == 2:
                np_image = np.stack((np_image,) * 3, axis=-1)
            if self.return_image:
                observation = np_image
            else:
                image = (
                    self.preprocess(Image.fromarray(np_image, "RGB"))
                    .unsqueeze(0)
                    .to(self.device)
                )
                image_features = self.model.encode_image(image)
                observation = image_features.reshape((-1)).detach().cpu().numpy()
                assert observation.shape == (self.emb_size,), print(
                    "obs shape: ", observation.shape
                )
        else:
            observation = np.zeros((self.emb_size,), dtype="float32")

        self.cnt = 0
        self.state = idx

        return observation, info

    def test_sequential(self, model, dataset="imagenet"):
        cumulative_reward = []
        action_list = {model_name: 0 for model_name in self.bandits.values()}
        for i in range(val_set_size):
            print(f"step {i } ") if i % 2000 == 0 else None

            obs, info = self.reset(seed=42, _idx=i)
            action, _states = model.predict(obs, deterministic=True)
            action = action.item(0)
            obs, reward, terminated, truncated, info = self.step(action)
            cumulative_reward.append(reward)
            action_list[self.bandits[action]] += 1
            if terminated:
                self.reset()

        self.close()
        cumulative_reward = np.cumsum(cumulative_reward)
        print(f"cumulative reward: {cumulative_reward[-1]}")
        print(f"action list: {action_list}")

        return cumulative_reward
