import argparse
import os
import pickle

import clip
import torch
import torchvision
from PIL import Image

torchvision.__version__

parser = argparse.ArgumentParser(description="Create Configuration")
parser.add_argument("-d", "--device", type=int, help="device name", default=-1)
args = parser.parse_args()
print(args)

device = (
    f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
)

### Load the floor plan png file
alfred_floor_path = "./synced_data/csv/alfred_data/floorplan/"
# iterate through the floor plan png files
floor_plan_list = {}
for file in os.listdir(alfred_floor_path):
    if file.endswith(".png"):
        floor_plan_list[file] = (
            Image.open(alfred_floor_path + file).convert("RGB").resize((336, 336))
        )

### Load CLIP model
model, preprocess = clip.load(
    "ViT-L/14@336px", jit=False, device=device
)  # ViT-L/14@336px, ViT-B/32

### encode the floor plan png files
floor_plan_list_emb = {}
for floor_plan in floor_plan_list.keys():
    floor_plan_list_emb[floor_plan] = (
        model.encode_image(
            preprocess(floor_plan_list[floor_plan]).unsqueeze(0).to(device)
        )
        .cpu()
        .detach()
        .numpy()
    )

### save floor_plan_list_emb
with open("synced_data/csv/alfred_data/floor_plan.pkl", "wb") as f:
    pickle.dump(floor_plan_list_emb, f)
    print("floor_plan_list_emb saved", len(floor_plan_list_emb))
