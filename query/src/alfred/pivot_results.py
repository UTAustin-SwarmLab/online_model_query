import pandas as pd

df = pd.read_csv("synced_data/csv/alfred_data/alfred_models_results.csv")
df.sort_values(by=["task_type"], inplace=True)

df = df.pivot(index=["task_idx", "repeat_idx"], columns=["model"], values="SR")
df["remote_model"] = df[["HLSM", "HiTUT"]].max(axis=1)

total_cnt = len(df)
print(f"Total count: {total_cnt}")
df["remote_local"] = df["remote_model"] - df["FILM"]
local_remote_cnt = (df["remote_local"] == -1.0).sum()
remote_local_cnt = (df["remote_local"] == 1.0).sum()


df["remote+local"] = df["remote_model"] + df["FILM"]
print(f"Local-Remote count: {local_remote_cnt}")
print(f"Remote-Local count: {remote_local_cnt}")
print(f"Remote nor Local count: {(df['remote+local'] == 0.0).sum()}")
print(f"Remote and Local count: {(df['remote+local'] == 2.0).sum()}")

### print with ratio
print(f"Local-Remote ratio: {local_remote_cnt/total_cnt}")
print(f"Remote-Local ratio: {remote_local_cnt/total_cnt}")
print(f"Remote nor Local ratio: {(df['remote+local'] == 0.0).sum()/total_cnt}")
print(f"Remote and Local ratio: {(df['remote+local'] == 2.0).sum()/total_cnt}")

df.drop(columns=["HLSM", "HiTUT"], inplace=True)
df.to_csv("synced_data/csv/alfred_data/alfred_models_results_pivot.csv")
