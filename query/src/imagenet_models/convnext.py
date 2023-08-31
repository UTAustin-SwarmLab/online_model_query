import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification

device = "cuda:0" if torch.cuda.is_available() else "cpu"

inference_number = 50000

### load model
dataset = load_dataset(
    "imagenet-1k", split="validation", streaming=False, use_auth_token=True
).with_format("pt")
processor = AutoImageProcessor.from_pretrained(
    "facebook/convnextv2-tiny-22k-384", device=device
)
model = ConvNextV2ForImageClassification.from_pretrained(
    "facebook/convnextv2-tiny-22k-384"
).to(device)

### preparing dataframe
df1k = []
for id in range(1000):
    df1k.append([id, model.config.id2label[id], 0, 0, "convnext"])
df1k = pd.DataFrame(
    df1k, columns=["read_id", "real_label", "pred_wrong", "pred_correct", "model"]
)

df1k_pred = []
for id in range(1000):
    df1k_pred.append([id, model.config.id2label[id], 0, 0, "convnext"])
df1k_pred = pd.DataFrame(
    df1k_pred, columns=["pred_id", "pred_label", "pred_wrong", "pred_correct", "model"]
)

validation_data = []

### inference
for i in range(inference_number):
    image, label = dataset[i]["image"], dataset[i]["label"].item()
    ### convert grayscale to RGB
    if len(image.shape) == 2:
        image = image.unsqueeze(-1).repeat(1, 1, 3)
    ### print process
    if i % 1000 == 0:
        print(i, image.shape, label)

    inputs = processor(image, return_tensors="pt").to(
        device
    )  # {'pixel_values': tensor torch.Size([1, 3, 224, 224])}

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    pred_label = logits.argmax(-1).item()
    validation_data.append(
        [
            label,
            model.config.id2label[label],
            pred_label,
            model.config.id2label[pred_label],
            pred_label == label,
            "convnext",
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
df_validation.to_csv(
    "./synced_data/csv/imagenet/convnext_df_validation.csv", index=False
)

df1k.to_csv("./synced_data/csv/imagenet/convnext_df1k.csv", index=False)
df1k_pred.to_csv("./synced_data/csv/imagenet/convnext_df1k_pred.csv", index=False)
