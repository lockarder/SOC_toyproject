import os
import joblib
import torch
import optuna
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import batch_convert_mat_to_csv, clean_soc_csv_files, load_all_clean_csvs
from dataset import BatteryDatasetLSTM
from mymodel import BasicLSTM

# ==== 路径配置 ====
mat_dir = os.path.join("data", "raw", "batterydata")
processed_dir = os.path.join("data", "processed")
scaler_save_path = os.path.join("outputs", "scaler.save")
os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)

feature_cols = ['Delta_t', 'Voltage', 'Current', 'Temperature']
label_col = 'SOC'

# ==== 数据处理 ====
batch_convert_mat_to_csv(mat_dir, processed_dir)
clean_soc_csv_files(processed_dir)
df = load_all_clean_csvs(processed_dir)

print(f"总样本数: {len(df)}")
print(f"涉及文件: {df['SourceFile'].unique()}")

# ==== 划分周期 ====
unique_cycles = df[['SourceFile', 'Cycle_Index']].drop_duplicates()
train_cycles, temp_cycles = train_test_split(unique_cycles, test_size=0.3, random_state=42)
val_cycles, test_cycles = train_test_split(temp_cycles, test_size=0.5, random_state=42)

def filter_cycles(data, cycles):
    return data.merge(cycles, on=['SourceFile', 'Cycle_Index'], how='inner')

train_df = filter_cycles(df, train_cycles)
val_df = filter_cycles(df, val_cycles)
test_df = filter_cycles(df, test_cycles)

print(f"Train set: {len(train_df)}, Val set: {len(val_df)}, Test set: {len(test_df)}")

# ==== 标准化 ====
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

# ==== 固定参数 ====
input_size = len(feature_cols)
sequence_length = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== 评估函数 ====
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples

# ==== Optuna 调参目标函数 ====
def objective(trial):
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_dataset = BatteryDatasetLSTM(train_df_scaled, feature_cols, label_col, sequence_length)
    val_dataset = BatteryDatasetLSTM(val_df_scaled, feature_cols, label_col, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BasicLSTM(input_size, hidden_size, num_layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience, trigger_times = 3, 0
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        val_loss = evaluate(model, val_loader, criterion, device)

        # 打印当前 trial 和 epoch 信息
        print(f"[Trial {trial.number}] Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        trial.report(val_loss, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch+1}")
            raise optuna.exceptions.TrialPruned()

        # Early stopping
        if val_loss + 1e-5 < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} for trial {trial.number}")
                break

    return best_val_loss

# ==== 执行调参 ====
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best trial:")
print(study.best_trial.params)

# ==== Step 6: 使用最优超参定义模型 ====
best_params = study.best_trial.params
hidden_size = best_params["hidden_size"]
num_layers = best_params["num_layers"]
dropout = best_params["dropout"]
lr = best_params["lr"]
batch_size = best_params["batch_size"]

print(f"Best hyperparameters: {best_params}")

# ==== Step 7: 构建最终训练用的数据 ====
train_dataset = BatteryDatasetLSTM(train_df_scaled, feature_cols, label_col, sequence_length)
val_dataset = BatteryDatasetLSTM(val_df_scaled, feature_cols, label_col, sequence_length)
test_dataset = BatteryDatasetLSTM(test_df_scaled, feature_cols, label_col, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==== Step 8: 模型训练准备 ====
input_size = len(feature_cols)
model = BasicLSTM(input_size, hidden_size, num_layers, dropout).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

num_epochs = 50
best_val_loss = float('inf')
patience = 5
trigger_times = 0
min_delta = 1e-5  # 最小改善幅度

# ==== Step 9: 训练过程 ====
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples

print("Start final training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)

    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # 早停逻辑
    if val_loss + min_delta < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), "outputs/best_model.pth")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# ==== Step 10: 加载最优模型，评估测试集 ====
model.load_state_dict(torch.load("outputs/best_model.pth"))
test_loss = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.6f}")

# ==== 保存最终结果 ====
with open("outputs/final_results.txt", "w") as f:
    f.write(f"Best Params: {best_params}\n")
    f.write(f"Test Loss: {test_loss:.6f}\n")

print("Final training complete.")