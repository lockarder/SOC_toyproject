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
from utils import batch_convert_mat_to_csv, clean_soc_csv_files, load_all_clean_csvs
from dataset import BatteryDatasetLSTM
from mymodel import BasicLSTM, Basic1DCNN
from utils import get_device

# 固定随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==== 训练和评估函数 ==== 
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device).float()
        labels = labels.to(device).float().unsqueeze(-1)  # 保证标签shape是 (batch,1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float().unsqueeze(-1)  # 保证标签shape是 (batch,1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples

# ==== 路径配置和目录创建 ==== 
base_output_dir = os.path.join("outputs", "Basic1DCNN")
os.makedirs(base_output_dir, exist_ok=True)

mat_dir = os.path.join("data", "raw", "batterydata")
processed_dir = os.path.join("data", "processed")
scaler_save_path = os.path.join(base_output_dir, "scaler.save")
os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)

feature_cols = ['Delta_t', 'Voltage', 'Current', 'Temperature']
label_col = 'SOC'

# ==== 数据处理 ====
batch_convert_mat_to_csv(mat_dir, processed_dir)
clean_soc_csv_files(processed_dir)
df = load_all_clean_csvs(processed_dir)

print(f"总样本数: {len(df)}")
print(f"涉及文件: {df['SourceFile'].unique()}")

unique_cycles = df[['SourceFile', 'Cycle_Index']].drop_duplicates()
train_cycles, temp_cycles = train_test_split(unique_cycles, test_size=0.3, random_state=42)
val_cycles, test_cycles = train_test_split(temp_cycles, test_size=0.5, random_state=42)

def filter_cycles(data, cycles):
    return data.merge(cycles, on=['SourceFile', 'Cycle_Index'], how='inner')

train_df = filter_cycles(df, train_cycles)
val_df = filter_cycles(df, val_cycles)
test_df = filter_cycles(df, test_cycles)

print(f"Train set: {len(train_df)}, Val set: {len(val_df)}, Test set: {len(test_df)}")

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
sequence_length = 30  # 统一序列长度

# ==== Optuna调参目标函数 ====
def objective(trial):
    num_filters = trial.suggest_categorical("num_filters", [32, 64, 128])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    stride = trial.suggest_int("stride", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    pool_type = trial.suggest_categorical("pool_type", ["avg", "max"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_dataset = BatteryDatasetLSTM(train_df_scaled, feature_cols, label_col, sequence_length)
    val_dataset = BatteryDatasetLSTM(val_df_scaled, feature_cols, label_col, sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Basic1DCNN(
        input_size=len(feature_cols),
        num_filters=num_filters,
        kernel_size=kernel_size,
        num_layers=num_layers,
        stride=stride,
        pool_type=pool_type,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience, trigger_times = 3, 0
    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

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

# ==== 执行调参 ====
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best trial params:")
print(study.best_trial.params)

# ==== 使用最优超参准备数据和模型 ====
best_params = study.best_trial.params

train_dataset = BatteryDatasetLSTM(train_df_scaled, feature_cols, label_col, sequence_length)
val_dataset = BatteryDatasetLSTM(val_df_scaled, feature_cols, label_col, sequence_length)
test_dataset = BatteryDatasetLSTM(test_df_scaled, feature_cols, label_col, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)

model = Basic1DCNN(
    input_size=len(feature_cols),
    num_filters=best_params["num_filters"],
    kernel_size=best_params["kernel_size"],
    num_layers=best_params["num_layers"],
    stride=best_params["stride"],
    pool_type=best_params["pool_type"],
    dropout=best_params["dropout"]
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
model.load_state_dict(torch.load(os.path.join(base_output_dir, "best_model.pth")))
test_loss = evaluate(model, test_loader, criterion, device)
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
