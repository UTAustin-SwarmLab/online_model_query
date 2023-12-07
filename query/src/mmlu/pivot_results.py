import pandas as pd

bandits = {
    0: "vicuna-7b-v1.5",
    # 1: "falcon-180B",
    2: "falcon-180B-chat",
    # 3: "qCammel-70-x",
    4: "Llama-2-70b-instruct",
    # 5: "Llama-2-70b-instruct-v2",
    6: "StableBeluga-13B",
    # 7: "airoboros-l2-70b",
}
# subtasks = {
#     "hendrycksTest-high_school_chemistry",
#     "hendrycksTest-high_school_computer_science",
#     "hendrycksTest-high_school_macroeconomics",
#     "hendrycksTest-high_school_government_and_politics",
# }

dfs = {}
for model in bandits.values():
    df = pd.read_csv(f"synced_data/csv/mmlu/{model}_nochoice.csv")
    # filter df by subdataset
    # idx = df["subdataset"].isin(subtasks)
    # df = df[idx]
    dfs[model] = df

df = pd.concat(dfs.values(), axis=0)
df.drop(columns=["gold", "acc", "example"], inplace=True)
df[df.duplicated(keep=False)]
df.sort_values(by=["Unnamed: 0"], inplace=True)
# df.set_index(["example", "Unnamed: 0"], inplace=True)
df.drop_duplicates(inplace=True)
print(df.head())


df = df.pivot(columns=["model"], values="acc_norm")
df["remote_model"] = df[
    ["falcon-180B-chat", "Llama-2-70b-instruct", "StableBeluga-13B"]
].max(axis=1)
print(df.head())


total_cnt = len(df)
print(f"Total count: {total_cnt}")
df["remote_local"] = df["remote_model"] - df["vicuna-7b-v1.5"]
local_remote_cnt = (df["remote_local"] == -1.0).sum()
remote_local_cnt = (df["remote_local"] == 1.0).sum()


df["remote+local"] = df["remote_model"] + df["vicuna-7b-v1.5"]
print(f"Local-Remote count: {local_remote_cnt}")
print(f"Remote-Local count: {remote_local_cnt}")
print(f"Remote nor Local count: {(df['remote+local'] == 0.0).sum()}")
print(f"Remote and Local count: {(df['remote+local'] == 2.0).sum()}")

### print with ratio
print(f"Local-Remote ratio: {local_remote_cnt/total_cnt}")
print(f"Remote-Local ratio: {remote_local_cnt/total_cnt}")
print(f"Remote nor Local ratio: {(df['remote+local'] == 0.0).sum()/total_cnt}")
print(f"Remote and Local ratio: {(df['remote+local'] == 2.0).sum()/total_cnt}")

df.drop(
    columns=["falcon-180B-chat", "Llama-2-70b-instruct", "StableBeluga-13B"],
    inplace=True,
)
df.to_csv("synced_data/csv/mmlu/mmlu_models_results_pivot.csv")
