import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import batch_convert_mat_to_csv, clean_soc_csv_files, load_all_clean_csvs
from mymodel import BasicLSTM
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BatteryDatasetLSTM
import optuna

# ==== 配置参数 ====
mat_dir = os.path.join("data", "raw", "batterydata")
processed_dir = os.path.join("data", "processed")
scaler_save_path = os.path.join("outputs", "scaler.save")

os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)

feature_cols = ['Delta_t', 'Voltage', 'Current', 'Temperature']
label_col = 'SOC'

# ==== Step 1: 数据转换 ====
batch_convert_mat_to_csv(mat_dir, processed_dir)
clean_soc_csv_files(processed_dir)
df = load_all_clean_csvs(processed_dir)

print(f"总样本数: {len(df)}")
print(f"涉及文件: {df['SourceFile'].unique()}")

# ==== Step 2: 划分周期 ====
unique_cycles = df[['SourceFile', 'Cycle_Index']].drop_duplicates()
train_cycles, temp_cycles = train_test_split(unique_cycles, test_size=0.3, random_state=42)
val_cycles, test_cycles = train_test_split(temp_cycles, test_size=0.5, random_state=42)

def filter_cycles(data, cycles):
    return data.merge(cycles, on=['SourceFile', 'Cycle_Index'], how='inner')

train_df = filter_cycles(df, train_cycles)
val_df = filter_cycles(df, val_cycles)
test_df = filter_cycles(df, test_cycles)

print(f"Train set: {len(train_df)}, Val set: {len(val_df)}, Test set: {len(test_df)}")

# ==== Step 3: 标准化 ====
def standardize_features(train_df, val_df, test_df, feature_cols, save_path):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    joblib.dump(scaler, save_path)

    train_scaled = scaler.transform(train_df[feature_cols])
    val_scaled = scaler.transform(val_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    return train_scaled, val_scaled, test_scaled

train_features_scaled, val_features_scaled, test_features_scaled = standardize_features(
    train_df, val_df, test_df, feature_cols, scaler_save_path)

# ==== Step 4: 把标准化后的特征替换回DataFrame ====
def replace_features_in_df(original_df, scaled_features, feature_cols):
    df_copy = original_df.copy()
    df_copy[feature_cols] = scaled_features
    return df_copy

train_df_scaled = replace_features_in_df(train_df, train_features_scaled, feature_cols)
val_df_scaled = replace_features_in_df(val_df, val_features_scaled, feature_cols)
test_df_scaled = replace_features_in_df(test_df, test_features_scaled, feature_cols)

# ==== Step 5: 构建 Dataset 和 DataLoader ====
sequence_length = 20
batch_size = 64

train_dataset = BatteryDatasetLSTM(train_df_scaled, feature_cols, label_col, sequence_length)
val_dataset = BatteryDatasetLSTM(val_df_scaled, feature_cols, label_col, sequence_length)
test_dataset = BatteryDatasetLSTM(test_df_scaled, feature_cols, label_col, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==== Step 6: 模型定义和训练准备 ====
input_size = len(feature_cols)  # 4
hidden_size = 64
num_layers = 2
dropout = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = BasicLSTM(input_size, hidden_size, num_layers, dropout).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ==== Step 7: 简单训练循环示例 ====
num_epochs = 10
best_val_loss = float('inf')
patience = 3
trigger_times = 0
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss

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
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "outputs/best_model.pth")  # 保存最优模型
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break
