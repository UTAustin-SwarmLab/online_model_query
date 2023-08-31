### python ./src/cifar100_models/pytorch-cifar-models.py -n resnet56 -d 0
import argparse
import json

import pandas as pd
import torch
from datasets import load_dataset

parser = argparse.ArgumentParser(description="Create Configuration")
parser.add_argument("-n", "--modelname", type=str, help="model name", default="")
parser.add_argument("-d", "--device", type=int, help="device name", default=-1)
args = parser.parse_args()


def batch_images(dataset, batch_size):
    images = []
    labels = []
    for i in range(batch_size):
        image, label = dataset[i]["img"], dataset[i]["fine_label"].item()
        images.append(image)
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)


device = (
    f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
)
inference_number = 10000

### load model
model_name = "cifar100_" + args.modelname  ### resnet56, mobilenetv2_x1_4, repvgg_a2
model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", model_name, pretrained=True
).to(device)
model.eval()
batch_size = 64

### load label mapping
with open("./src/cifar100_models/label_mapping.json") as f:
    id_mapping = json.load(f)

id_mapping = {int(k): v for k, v in id_mapping.items()}

### load dataset
dataset = load_dataset(
    "cifar100", split="test", streaming=False, use_auth_token=True
).with_format("pt")
dataset = dataset.map(batched=True)

### preparing dataframe
dfcifar = []
for id in range(100):
    dfcifar.append([id, id_mapping[id], 0, 0, model_name.split("_", 1)[1]])
dfcifar = pd.DataFrame(
    dfcifar, columns=["read_id", "real_label", "pred_wrong", "pred_correct", "model"]
)

dfcifar_pred = []
for id in range(100):
    dfcifar_pred.append([id, id_mapping[id], 0, 0, model_name.split("_", 1)[1]])
dfcifar_pred = pd.DataFrame(
    dfcifar_pred,
    columns=["pred_id", "pred_label", "pred_wrong", "pred_correct", "model"],
)

validation_data = []

### inference
cnt = 0
correct = 0
for i in range(inference_number):
    image, label = dataset[i]["img"], dataset[i]["fine_label"].item()
    image = image.permute(2, 0, 1).to(device)  # from HWC to CHW: 32x32x3 to 3x32x32
    image = image / 255.0  # normalize to [0, 1]
    ### normalize image
    """
    mean: [0.5070, 0.4865, 0.4409], std: [0.2673, 0.2564, 0.2761]
    """
    image[0] = (image[0] - 0.5070) / 0.2673
    image[1] = (image[1] - 0.4865) / 0.2564
    image[2] = (image[2] - 0.4409) / 0.2761

    with torch.no_grad():
        outputs = model(image.unsqueeze(0))

    # model predicts one of the 100 cifar classes
    pred_label = outputs.argmax(-1).item()
    # print(pred_label, id_mapping[pred_label])
    # print(label, id_mapping[label])
    validation_data.append(
        [
            label,
            id_mapping[label],
            pred_label,
            id_mapping[pred_label],
            pred_label == label,
            model_name.split("_", 1)[1],
        ]
    )
    if pred_label != label:  # wrong prediction
        # print(f"Wrong prediction: {pred_label} {id_mapping[pred_label]} \t GT: {label} {id_mapping[label]}")
        dfcifar.loc[dfcifar["read_id"] == label, "pred_wrong"] += 1
        dfcifar_pred.loc[dfcifar_pred["pred_id"] == pred_label, "pred_wrong"] += 1
    else:  # correct prediction
        correct += 1
        dfcifar.loc[dfcifar["read_id"] == label, "pred_correct"] += 1
        dfcifar_pred.loc[dfcifar_pred["pred_id"] == pred_label, "pred_correct"] += 1

    ### print process
    cnt += 1
    if i % 1000 == 0:
        print(correct / cnt)
        print(i, image.shape, label)

print("Final accuracy:", correct / cnt)

df_validation = pd.DataFrame(
    validation_data,
    columns=["real_id", "real_label", "pred_id", "pred_label", "correctTF", "model"],
)
df_validation.to_csv(
    f'./synced_data/csv/cifar100/{model_name.split("_", 1)[1]}_df_validation.csv',
    index=False,
)

dfcifar.to_csv(
    f'./synced_data/csv/cifar100/{model_name.split("_", 1)[1]}_df1k.csv', index=False
)
dfcifar_pred.to_csv(
    f'./synced_data/csv/cifar100/{model_name.split("_", 1)[1]}_df1k_pred.csv',
    index=False,
)
