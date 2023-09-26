### convert csv files to np arrays
import numpy as np
import pandas as pd

imagenet_val_set_size = 50000
cifar100_val_set_size = 10000

bandits = {
    0: ("imagenet-1k", "convnext"),
    1: ("imagenet-1k", "mit"),
    2: ("imagenet-1k", "mobilenet"),
    3: ("imagenet-1k", "resnet"),
    4: ("imagenet-1k", "ViT"),
    5: ("cifar100", "mobilenetv2_x1_4"),
    6: ("cifar100", "repvgg_a2"),
    7: ("cifar100", "resnet56"),
}


dfs = []
for idx, values in bandits.items():
    dataset, model = values
    csv_path = f"./synced_data/csv/{dataset}/{model}_df_validation.csv"
    df = pd.read_csv(csv_path)
    dfs.append(df)
    print(idx, dataset, model, df.shape)

### cifar100
cifar_np = np.zeros((cifar100_val_set_size, 8))
for i in range(cifar100_val_set_size):
    for idx, df in enumerate(dfs[5:]):
        idx += 5
        correctTF = df.iloc[i]["correctTF"]
        cifar_np[i, idx] = 1 if correctTF else 0

print(cifar_np.shape)
print(np.sum(cifar_np[:, :5], axis=0))
print(np.sum(cifar_np[:, 5:], axis=0))
np.save("./synced_data/csv/cifar100/cifar100_val.npy", cifar_np)

### imagenet-1k
imagenet_np = np.zeros((imagenet_val_set_size, 8))
for i in range(imagenet_val_set_size):
    for idx, df in enumerate(dfs[:5]):
        correctTF = df.iloc[i]["correctTF"]
        imagenet_np[i, idx] = 1 if correctTF else 0

print(imagenet_np.shape)
print(imagenet_np)
print(np.sum(imagenet_np[:, :5], axis=0))
print(np.sum(imagenet_np[:, 5:], axis=0))
np.save("./synced_data/csv/imagenet-1k/imagenet-1k_val.npy", imagenet_np)
