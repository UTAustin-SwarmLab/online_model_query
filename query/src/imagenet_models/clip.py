import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor, ViTForImageClassification

# device = "cpu"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

inference_number = 50000

### load model
dataset = load_dataset(
    "imagenet-1k", split="validation", streaming=False, use_auth_token=True
).with_format("numpy")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", device=device)
model_hf = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(
    device
)

### preparing dataframe
df1k = []
for id in range(1000):
    df1k.append([id, model_hf.config.id2label[id], 0, 0, "clip"])
df1k = pd.DataFrame(
    df1k, columns=["read_id", "real_label", "pred_wrong", "pred_correct", "model"]
)

df1k_pred = []
for id in range(1000):
    df1k_pred.append([id, model_hf.config.id2label[id], 0, 0, "clip"])
df1k_pred = pd.DataFrame(
    df1k_pred, columns=["pred_id", "pred_label", "pred_wrong", "pred_correct", "model"]
)

validation_data = []

### get labels
labels = []
for id in range(0, 1000):
    labels.append((id, model_hf.config.id2label[id].split(",")[0]))
print("labels: ", labels[:5])
texts = [label[1] for label in labels]
print("text: ", texts[:5])

### inference
for i in range(inference_number):
    np_image, label = dataset[i]["image"], dataset[i]["label"].item()
    ### print process
    if i % 100 == 0:
        print(i, np_image.shape, label)
    if len(np_image.shape) == 2:
        np_image = np.stack((np_image,) * 3, axis=-1)
    inputs = processor(
        text=texts, images=np_image, return_tensors="pt", padding=True
    ).to(device)
    outputs = model(**inputs)
    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    probs = logits_per_image.softmax(
        dim=1
    )  # we can take the softmax to get the label probabilities

    # model predicts one of the 1000 ImageNet classes
    pred_label = probs.argmax(-1).item()
    validation_data.append(
        [
            label,
            model_hf.config.id2label[label],
            pred_label,
            model_hf.config.id2label[pred_label],
            pred_label == label,
            "clip",
        ]
    )
    if pred_label != label:  # wrong prediction
        df1k.loc[df1k["read_id"] == label, "pred_wrong"] += 1
        df1k_pred.loc[df1k_pred["pred_id"] == pred_label, "pred_wrong"] += 1
    else:  # correct prediction
        df1k.loc[df1k["read_id"] == label, "pred_correct"] += 1
        df1k_pred.loc[df1k_pred["pred_id"] == pred_label, "pred_correct"] += 1

df_validation = pd.DataFrame(
    validation_data,
    columns=["real_id", "real_label", "pred_id", "pred_label", "correctTF", "model"],
)
df_validation.to_csv("./csv/imagenet/clip_df_validation.csv", index=False)

df1k.to_csv("./csv/imagenet/clip_df1k.csv", index=False)
df1k_pred.to_csv("./csv/imagenet/clip_df1k_pred.csv", index=False)
