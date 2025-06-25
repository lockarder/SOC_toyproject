import os
import random
import joblib
import torch
import optuna
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 你自己写的工具函数和数据集类
from utils import batch_convert_mat_to_csv, clean_soc_csv_files, load_all_clean_csvs, get_device
from dataset import BatteryDatasetLSTM
from mymodel import BasicLSTM, Basic1DCNN, BasicCNNLSTMParallel  # 并行模型

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 训练函数
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device).float()
        labels = labels.to(device).float().unsqueeze(-1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

# 验证/测试函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float().unsqueeze(-1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples

# ==== 路径配置 ====
base_output_dir = os.path.join("outputs", "BasicCNNLSTMParallel")
os.makedirs(base_output_dir, exist_ok=True)

mat_dir = os.path.join("data", "raw", "batterydata")
processed_dir = os.path.join("data", "processed")
scaler_save_path = os.path.join(base_output_dir, "scaler.save")
os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)

feature_cols = ['Delta_t', 'Voltage', 'Current', 'Temperature']
label_col = 'SOC'

# ==== 数据准备 ====
batch_convert_mat_to_csv(mat_dir, processed_dir)
clean_soc_csv_files(processed_dir)
df = load_all_clean_csvs(processed_dir)

unique_cycles = df[['SourceFile', 'Cycle_Index']].drop_duplicates()
train_cycles, temp_cycles = train_test_split(unique_cycles, test_size=0.3, random_state=42)
val_cycles, test_cycles = train_test_split(temp_cycles, test_size=0.5, random_state=42)

def filter_cycles(data, cycles):
    return data.merge(cycles, on=['SourceFile', 'Cycle_Index'], how='inner')

train_df = filter_cycles(df, train_cycles)
val_df = filter_cycles(df, val_cycles)
test_df = filter_cycles(df, test_cycles)

# 标准化
def standardize_features(train_df, val_df, test_df, feature_cols, save_path):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    joblib.dump(scaler, save_path)
    return scaler.transform(train_df[feature_cols]), scaler.transform(val_df[feature_cols]), scaler.transform(test_df[feature_cols])

train_features_scaled, val_features_scaled, test_features_scaled = standardize_features(
    train_df, val_df, test_df, feature_cols, scaler_save_path)

def replace_features_in_df(original_df, scaled_features, feature_cols):
    df_copy = original_df.copy()
    df_copy[feature_cols] = scaled_features
    return df_copy

train_df_scaled = replace_features_in_df(train_df, train_features_scaled, feature_cols)
val_df_scaled = replace_features_in_df(val_df, val_features_scaled, feature_cols)
test_df_scaled = replace_features_in_df(test_df, test_features_scaled, feature_cols)

device = get_device()
sequence_length = 30

# ==== Optuna调参目标函数 ====
def objective(trial):
    cnn_params = {
        "num_filters": trial.suggest_categorical("num_filters", [32, 64, 128]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "num_layers": trial.suggest_int("cnn_num_layers", 1, 4),
        "stride": trial.suggest_int("stride", 1, 3),
        "pool_type": trial.suggest_categorical("pool_type", ["avg", "max"]),
        "dropout": trial.suggest_float("cnn_dropout", 0.0, 0.5)
    }

    lstm_params = {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_int("lstm_num_layers", 1, 3),
        "dropout": trial.suggest_float("lstm_dropout", 0.0, 0.5)
    }

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    fc_hidden_size = trial.suggest_categorical("fc_hidden_size", [32, 64, 128])
    fusion_dropout = trial.suggest_float("fusion_dropout", 0.0, 0.5)

    train_dataset = BatteryDatasetLSTM(train_df_scaled, feature_cols, label_col, sequence_length)
    val_dataset = BatteryDatasetLSTM(val_df_scaled, feature_cols, label_col, sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BasicCNNLSTMParallel(
        input_size=len(feature_cols),
        cnn_params=cnn_params,
        lstm_params=lstm_params,
        fc_hidden_size=fc_hidden_size,
        dropout=fusion_dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience, trigger_times = 3, 0
    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"[Trial {trial.number}] Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss + 1e-5 < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break

    return best_val_loss

# ==== 运行调参 ====
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best trial params:")
print(study.best_trial.params)

# ==== 用最优参数训练最终模型 ====

best_params = study.best_trial.params

train_dataset = BatteryDatasetLSTM(train_df_scaled, feature_cols, label_col, sequence_length)
val_dataset = BatteryDatasetLSTM(val_df_scaled, feature_cols, label_col, sequence_length)
test_dataset = BatteryDatasetLSTM(test_df_scaled, feature_cols, label_col, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)

# 创建模型
model = BasicCNNLSTMParallel(
    input_size=len(feature_cols),
    cnn_params={
        "num_filters": best_params["num_filters"],
        "kernel_size": best_params["kernel_size"],
        "num_layers": best_params["cnn_num_layers"],
        "stride": best_params["stride"],
        "pool_type": best_params["pool_type"],
        "dropout": best_params["cnn_dropout"]
    },
    lstm_params={
        "hidden_size": best_params["hidden_size"],
        "num_layers": best_params["lstm_num_layers"],
        "dropout": best_params["lstm_dropout"]
    },
    fc_hidden_size=best_params["fc_hidden_size"],
    dropout=best_params["fusion_dropout"]
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])

# ==== 训练和验证 ====
best_val_loss = float('inf')
trigger_times = 0
patience = 5
min_delta = 1e-5
num_epochs = 50

train_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = evaluate(model, val_loader, criterion, device)

    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    if val_loss + min_delta < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), os.path.join(base_output_dir, "best_model.pth"))
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# ==== 测试集评估 ====

# 注意这里要重新实例化模型，并加载最优参数，保证一致性
best_model = BasicCNNLSTMParallel(
    input_size=len(feature_cols),
    cnn_params={
        "num_filters": best_params["num_filters"],
        "kernel_size": best_params["kernel_size"],
        "num_layers": best_params["cnn_num_layers"],
        "stride": best_params["stride"],
        "pool_type": best_params["pool_type"],
        "dropout": best_params["cnn_dropout"]
    },
    lstm_params={
        "hidden_size": best_params["hidden_size"],
        "num_layers": best_params["lstm_num_layers"],
        "dropout": best_params["lstm_dropout"]
    },
    fc_hidden_size=best_params["fc_hidden_size"],
    dropout=best_params["fusion_dropout"]
).to(device)

best_model.load_state_dict(torch.load(os.path.join(base_output_dir, "best_model.pth")))
best_model.eval()

test_loss = evaluate(best_model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.6f}")

# ==== 保存结果 ====
with open(os.path.join(base_output_dir, "final_results.txt"), "w") as f:
    f.write(f"Best Params: {best_params}\n")
    f.write(f"Test Loss: {test_loss:.6f}\n")

# ==== 保存损失曲线 ====
loss_df = pd.DataFrame({
    "epoch": list(range(1, len(train_loss_list) + 1)),
    "train_loss": train_loss_list,
    "val_loss": val_loss_list
})
loss_df.to_csv(os.path.join(base_output_dir, "loss_history.csv"), index=False)

print("Final training complete.")

