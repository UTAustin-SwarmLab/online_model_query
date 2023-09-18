### ImageNet1kCIFAR100GymEnv class for gym 
import random
from typing import Tuple

import clip
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from gymnasium import spaces
from PIL import Image

imagenet_val_set_size = 50000
cifar100_val_set_size = 10000


class ImageNet1kCIFAR100GymEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_bandits: int = 8,
        h: int = 512,
        w: int = 512,
        c: int = 3,
        emb_size: int = 512,
        p: list[float] = [0.5, 0.5],
        device: str or torch.device = "cpu",
        return_image: bool = False,
        contextual: bool = True,
        replace_sample: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            n_bandits: number of arms of the bandit
            h: height of the image
            w: width of the image
            c: number of channels of the image
            emb_size: size of the embedding
            p: probability of selecting each dataset
            device: device to run the clip model
            return_image: whether to return image or embedding
            contextual: whether to use contextual bandit
            replace_sample: whether to replace the sample
        """
        super(ImageNet1kCIFAR100GymEnv, self).__init__()

        ### make sure the sum of p is 1
        assert sum(p) == 1, print("sum of p should be 1")
        assert len(p) == 2, print(
            "p should be a list of 2 elements (p_imagenet, p_cifar100)"
        )
        self.p = p
        ### Define action and observation space with discrete actions:
        self.action_space = spaces.Discrete(n_bandits)
        self.bandits = {
            0: ("imagenet-1k", "convnext"),
            1: ("imagenet-1k", "mit"),
            2: ("imagenet-1k", "mobilenet"),
            3: ("imagenet-1k", "resnet"),
            4: ("imagenet-1k", "ViT"),
            5: ("cifar100", "mobilenetv2_x1_4"),
            6: ("cifar100", "repvgg_a2"),
            7: ("cifar100", "resnet56"),
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
        for idx, values in self.bandits.items():
            dataset, model = values
            csv_path = f"./synced_data/csv/{dataset}/{model}_df_validation.csv"
            self.model_result_df[idx] = pd.read_csv(csv_path)

        ### input is an image or an embedding
        if return_image:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(h, w, c), dtype=np.uint8
            )
            self.h, self.w, self.c = h, w, c
        else:
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(emb_size,), dtype="float32"
            )
            self.model, self.preprocess = clip.load(
                "ViT-B/32", jit=False, device=self.device
            )

        self.imagenet = load_dataset(
            "imagenet-1k", split="validation", streaming=False, use_auth_token=True
        ).with_format("numpy")
        self.cifar100 = load_dataset(
            "cifar100", split="test", streaming=False, use_auth_token=True
        ).with_format("numpy")

        self.replace_sample = replace_sample
        if not self.replace_sample:
            self.imagenet_idx_seq = np.arange(imagenet_val_set_size)
            np.random.shuffle(self.imagenet_idx_seq)
            self.cifar100_idx_seq = np.arange(cifar100_val_set_size)
            np.random.shuffle(self.cifar100_idx_seq)

    def step(
        self, action: int, _idx: int = None, _dataset: str = None
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        '''
        Args:
            action: index of the action
            _idx: index of the next observation
            _dataset: name of the next dataset
        Returns:
            observation: next observation
            reward: reward
            terminated: whether the episode is terminated
            truncated: whether the episode is truncated
            info: info
        '''
        info = {}
        ### nothing is sequential here
        ### calculate reward
        selected_dataset, selected_model = self.bandits[action]  ### name of the model
        df_validation = self.model_result_df[action]  ### df of the model
        obs_dataset, current_idx = self.state
        if (selected_dataset == "imagenet-1k" and obs_dataset == "imagenet-1k") or (
            selected_dataset == "cifar100" and obs_dataset == "cifar100"
        ):
            correctTF = df_validation.iloc[current_idx]["correctTF"]
        else:
            correctTF = False
        reward = 1 if correctTF else 0

        ### calculate done
        self.cnt += 1
        terminated = False
        truncated = False

        ### choose the next dataset
        if _dataset is not None:
            next_dataset = _dataset
        else:
            next_dataset = np.random.choice(["imagenet-1k", "cifar100"], p=self.p)
        ### calculate next observation
        if next_dataset == "imagenet-1k":
            if self.replace_sample:
                idx = random.randint(0, imagenet_val_set_size - 1)
            ### force to select the next observation
            elif _idx is not None:
                idx = _idx
            else:
                ### pop an index from the list
                if self.imagenet_idx_seq.shape == (0,):
                    self.imagenet_idx_seq = np.arange(imagenet_val_set_size)
                    np.random.shuffle(self.imagenet_idx_seq)
                idx, self.imagenet_idx_seq = (
                    int(self.imagenet_idx_seq[-1]),
                    self.imagenet_idx_seq[:-1],
                )

            np_image, _ = (
                self.imagenet[idx]["image"],
                self.imagenet[idx]["label"].item(),
            )
        elif next_dataset == "cifar100":
            if self.replace_sample:
                idx = random.randint(0, cifar100_val_set_size - 1)
            ### force to select the next observation
            elif _idx is not None:
                idx = _idx
            else:
                ### pop an index from the list
                if self.cifar100_idx_seq.shape == (0,):
                    self.cifar100_idx_seq = np.arange(cifar100_val_set_size)
                    np.random.shuffle(self.cifar100_idx_seq)
                idx, self.cifar100_idx_seq = (
                    int(self.cifar100_idx_seq[-1]),
                    self.cifar100_idx_seq[:-1],
                )
            np_image, _ = (
                self.cifar100[idx]["img"],
                self.cifar100[idx]["fine_label"].item(),
            )

        if self.contextual:
            ### convert grayscale to RGB
            if len(np_image.shape) == 2:
                np_image = np.stack((np_image,) * 3, axis=-1)
            if self.return_image:
                import cv2

                observation = cv2.resize(
                    np_image, (self.h, self.w), interpolation=cv2.INTER_LINEAR
                )
            else:
                image = (
                    self.preprocess(Image.fromarray(np_image, "RGB"))
                    .unsqueeze(0)
                    .to(self.device)
                )
                image_features = self.model.encode_image(image)
                observation = (
                    image_features.reshape((-1))
                    .detach()
                    .cpu()
                    .numpy()
                    .astype("float32")
                )
                assert observation.shape == (self.emb_size,), print(
                    "obs shape: ", observation.shape
                )
        else:
            observation = np.zeros((self.emb_size,), dtype="float32")

        ### update next state
        self.state = next_dataset, idx  # update current idx

        if self.cnt % 2000 == 0:
            print(
                "cnt: ",
                self.cnt,
                "current state: ",
                (obs_dataset, current_idx),
                "action: ",
                self.bandits[action],
                "new state: ",
                self.state,
            )
            print(
                "reward: ", reward, "terminated: ", terminated, "truncated: ", truncated
            )

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: int = None,
        _idx: int = None,
        _dataset: str = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        '''
        Args:
            seed: random seed
            _idx: index of the next observation
            _dataset: name of the next dataset
        Returns:
            observation: next observation
            info: info
        '''
        super().reset(seed=seed, **kwargs)
        info = {}
        self.cnt = 0
        self.state = "_", -1
        observation, reward, terminated, truncated, info = self.step(
            0, _idx=_idx, _dataset=_dataset
        )

        return observation, info

    def test_sequential(self, model, dataset: str) -> Tuple[np.ndarray, list, dict]:
        '''
        Args:
            model: model to test
            dataset: dataset to test
        Returns:
            cumulative_reward: cumulative reward
            obs_list: list of observations
            action_list: list of actions
        '''
        cumulative_reward = []
        obs_list = []
        action_list = {model_name: 0 for model_name in self.bandits.values()}
        if dataset == "imagenet-1k":
            for i in range(imagenet_val_set_size):
                print(f"step {i } ") if i % 2000 == 0 else None
                obs, info = self.reset(seed=42, _idx=i, _dataset="imagenet-1k")
                action, _states = model.predict(obs, deterministic=True)
                action = action.item(0)
                obs, reward, terminated, truncated, info = self.step(action)
                cumulative_reward.append(reward)
                action_list[self.bandits[action]] += 1
                obs_list.append(obs)
                if terminated:
                    self.reset()
        elif dataset == "cifar100":
            for i in range(cifar100_val_set_size):
                print(f"step {i } ") if i % 2000 == 0 else None
                obs, info = self.reset(seed=42, _idx=i, _dataset="cifar100")
                action, _states = model.predict(obs, deterministic=True)
                action = action.item(0)
                obs, reward, terminated, truncated, info = self.step(action)
                cumulative_reward.append(reward)
                action_list[self.bandits[action]] += 1
                obs_list.append(obs)
                if terminated:
                    self.reset()
        else:
            raise NotImplementedError

        self.close()
        cumulative_reward = np.cumsum(cumulative_reward)
        print(f"Dataset: {dataset}")
        print(f"cumulative reward: {cumulative_reward[-1]}")
        print(f"action list: {action_list}")

        return cumulative_reward, obs_list, action_list


class ImageNet1kCIFAR100GymEnvNp(gym.Env):
    def __init__(
        self, n_bandits=8, emb_size=512, contextual=True, replace_sample=False, **kwargs
    ):
        """
        Args:
            n_bandits: number of arms of the bandit
            emb_size: size of the embedding
            contextual: whether to use contextual bandit
            replace_sample: whether to replace the sample
        """
        super(ImageNet1kCIFAR100GymEnvNp, self).__init__()

        ### Define action and observation space with discrete actions:
        self.action_space = spaces.Discrete(n_bandits)
        self.bandits = {
            0: ("imagenet-1k", "convnext"),
            1: ("imagenet-1k", "mit"),
            2: ("imagenet-1k", "mobilenet"),
            3: ("imagenet-1k", "resnet"),
            4: ("imagenet-1k", "ViT"),
            5: ("cifar100", "mobilenetv2_x1_4"),
            6: ("cifar100", "repvgg_a2"),
            7: ("cifar100", "resnet56"),
        }
        self.contextual = contextual
        self.model_result_df = {}
        assert len(self.bandits) == n_bandits, print(
            "n_bandits should be equal to the number of bandits"
        )

        self.emb_size = emb_size
        self.reward_range = (0, 1)
        self.replace_sample = replace_sample

        ### input is an image or an embedding
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(emb_size,), dtype="float32"
        )

    def step(self, action: int)-> Tuple[np.ndarray, float, bool, bool, dict]:
        '''
        Args:
            action: index of the action
        Returns:
            observation: next observation
            reward: reward
            terminated: whether the episode is terminated
            truncated: whether the episode is truncated
            info: info
        ''' 

        info = {}
        ### nothing is sequential here
        ### calculate reward
        cur_idx = self.cnt
        self.X[cur_idx, :]
        cur_label = self.y[cur_idx, :]
        correctTF = cur_label[action] == 1
        reward = 1 if correctTF else 0

        ### calculate done
        self.cnt += 1
        terminated = False
        truncated = False

        ### calculate next observation
        if self.contextual:
            next_idx = self.cnt
            observation = self.X[next_idx, :]
        else:
            observation = np.zeros((self.emb_size,), dtype="float32")

        if self.cnt % 2000 == 0:
            print("cnt: ", self.cnt, "action: ", self.bandits[action])
            print(
                "reward: ", reward, "terminated: ", terminated, "truncated: ", truncated
            )

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed=None,
        **kwargs,
    )-> Tuple[np.ndarray, dict]:
        '''
        Args:
            seed: random seed
        Returns:
            observation: next observation
            info: info
        '''
        super().reset(seed=seed, **kwargs)
        np.random.seed(seed)
        cifar_X = np.load("./synced_data/csv/clip_emb_cifar100.npy")
        imagenet_X = np.load("./synced_data/csv/clip_emb_imagenet-1k.npy")
        self.X = np.concatenate((cifar_X, imagenet_X), axis=0)
        arr = np.random.choice(
            np.arange(self.X.shape[0]),
            (imagenet_val_set_size + cifar100_val_set_size),
            replace=self.replace_sample,
        )
        self.X = self.X[arr, :]

        cifar_y = np.load("./synced_data/csv/cifar100/cifar100_val.npy")
        imagent_y = np.load("./synced_data/csv/imagenet-1k/imagenet-1k_val.npy")
        self.y = np.concatenate((cifar_y, imagent_y), axis=0)
        self.y = self.y[arr, :]

        info = {}
        self.cnt = 0
        observation, reward, terminated, truncated, info = self.step()

        return observation, info
