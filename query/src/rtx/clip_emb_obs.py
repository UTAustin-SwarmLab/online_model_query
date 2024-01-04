import argparse

import clip
import numpy as np
import torch
import torchvision
from PIL import Image

torchvision.__version__

parser = argparse.ArgumentParser(description="Create Configuration")
parser.add_argument("-d", "--device", type=int, help="device name", default=-1)
parser.add_argument(
    "-ds", "--dataset", type=str, help="dataset name", default="fractal20220817_data"
)
args = parser.parse_args()
print(args)

device = (
    f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
)

### Load the floor plan png file
rtx_obs_path = f"./synced_data/rtx/{args.dataset}_img.npy"
img_np = np.load(rtx_obs_path)
print(img_np.shape)  # (5000, 2, 256, 256, 3)

# iterate through the floor plan png files
obs_list = {}

### Load CLIP model
model, preprocess = clip.load(
    "ViT-L/14@336px", jit=False, device=device
)  # ViT-L/14@336px, ViT-B/32

### encode the floor plan png files
obs_emb_list = []
for idx in range(0, img_np.shape[0], 2):
    if idx % 100 == 0:
        print(idx)
    obs1 = img_np[idx]
    obs2 = img_np[idx + 1]

    PIL_img1 = Image.fromarray(np.uint8(obs1)).convert("RGB")
    PIL_img2 = Image.fromarray(np.uint8(obs2)).convert("RGB")
    emb_1 = (
        model.encode_image(preprocess(PIL_img1).unsqueeze(0).to(device))
        .cpu()
        .detach()
        .numpy()
    )
    emb_2 = (
        model.encode_image(preprocess(PIL_img2).unsqueeze(0).to(device))
        .cpu()
        .detach()
        .numpy()
    )
    emb = np.concatenate((emb_1, emb_2), axis=1).reshape(-1)  # (1536,)
    obs_emb_list.append(emb)

### save obs_list_emb
obs_emb_list = np.array(obs_emb_list)
np.save(f"./synced_data/csv/rtx/{args.dataset}_img_emb.npy", obs_emb_list)
