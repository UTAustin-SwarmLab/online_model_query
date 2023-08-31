import json

import pandas as pd
import torch
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTModel  # ViTFeatureExtractor

device = "cuda:0" if torch.cuda.is_available() else "cpu"
inference_number = 10000

### load label mapping
with open("./src/cifar100_models/label_mapping.json") as f:
    id_mapping = json.load(f)

id_mapping = {int(k): v for k, v in id_mapping.items()}

### load model
dataset = load_dataset(
    "cifar100", split="test", streaming=False, use_auth_token=True
).with_format("pt")
# processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k', device=device)
processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k", device=device
)
model = ViTModel.from_pretrained("edumunozsala/vit_base-224-in21k-ft-cifar100").to(
    device
)

### preparing dataframe
dfcifar = []
for id in range(100):
    dfcifar.append([id, id_mapping[id], 0, 0, "ViT"])
dfcifar = pd.DataFrame(
    dfcifar, columns=["read_id", "real_label", "pred_wrong", "pred_correct", "model"]
)

dfcifar_pred = []
for id in range(100):
    dfcifar_pred.append([id, id_mapping[id], 0, 0, "ViT"])
dfcifar_pred = pd.DataFrame(
    dfcifar_pred,
    columns=["pred_id", "pred_label", "pred_wrong", "pred_correct", "model"],
)

validation_data = []

### inference
for i in range(inference_number):
    image, label = dataset[i]["img"], dataset[i]["fine_label"].item()
    ### print process
    if i % 500 == 0:
        print(i, image.shape, label)
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        for key, value in outputs.items():
            print(key, value.shape)
        logits = outputs.logits

    # model predicts one of the 100 cifar classes
    pred_label = logits.argmax(-1).item()
    # print(pred_label, model.config.id2label[pred_label])
    # print(label, model.config.id2label[label])
    validation_data.append(
        [
            label,
            id_mapping[id],
            pred_label,
            id_mapping[pred_label],
            pred_label == label,
            "ViT",
        ]
    )
    if pred_label != label:  # wrong prediction
        print(
            f"Wrong prediction: {pred_label} {model.config.id2label[pred_label]} \t GT: {label} {model.config.id2label[label]}"
        )
        dfcifar.loc[dfcifar["read_id"] == label, "pred_wrong"] += 1
        dfcifar_pred.loc[dfcifar_pred["pred_id"] == pred_label, "pred_wrong"] += 1
    else:  # correct prediction
        dfcifar.loc[dfcifar["read_id"] == label, "pred_correct"] += 1
        dfcifar_pred.loc[dfcifar_pred["pred_id"] == pred_label, "pred_correct"] += 1

df_validation = pd.DataFrame(
    validation_data,
    columns=["real_id", "real_label", "pred_id", "pred_label", "correctTF", "model"],
)
df_validation.to_csv("./csv/ViT_df_validation.csv", index=False)

dfcifar.to_csv("./csv/ViT_df1k.csv", index=False)
dfcifar_pred.to_csv("./csv/ViT_df1k_pred.csv", index=False)
