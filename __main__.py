import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import plot_voltage_soc_by_cycle
from utils import parse_mat_to_df
from utils import batch_convert_mat_to_csv
from utils import clean_soc_csv_files
from utils import load_all_clean_csvs
from utils import plot_variable_curve
from sklearn.preprocessing import StandardScaler


mat_dir = r"data/raw/batterydata"
csv_dir = r"data/processed"

batch_convert_mat_to_csv(mat_dir, csv_dir)
csv_dir = "data/processed"
for filename in os.listdir(csv_dir):
    if filename.endswith("_soc.csv"):
        print(f"Found saved CSV file: {filename}")

csv_path = "data/processed/B0007_soc.csv"  # 修改成你想看的文件路径
df = pd.read_csv(csv_path)

print("CSV 文件的列标签:")
print(df.columns.tolist())
print(df.head())
#获得SOC-LAbel的索引 B0007_soc.csv

#plot_voltage_soc_by_cycle(r"data/processed/B0007_soc.csv")
processed_dir = "data/processed"
clean_soc_csv_files(processed_dir)
df = load_all_clean_csvs(processed_dir)

print(df.head())
print(f"总样本数: {len(df)}")
print(f"涉及文件: {df['SourceFile'].unique()}")
unique_cycles = df[['SourceFile', 'Cycle_Index']].drop_duplicates()
print(unique_cycles) #按照周期划分 
train_cycles, temp_cycles = train_test_split(unique_cycles, test_size=0.3, random_state=42)
val_cycles, test_cycles = train_test_split(temp_cycles, test_size=0.5, random_state=42)
print(df.columns)
def filter_cycles(data, cycles):
    return data.merge(cycles, on=['SourceFile', 'Cycle_Index'], how='inner')
#训练集 测试集 验证集 切割
train_df = filter_cycles(df, train_cycles)
val_df = filter_cycles(df, val_cycles)
test_df = filter_cycles(df, test_cycles)
#归一化处理
feature_cols = ['Delta_t', 'Voltage', 'Current', 'Temperature']

scaler = StandardScaler()
scaler.fit(train_df[feature_cols])  # 只用训练集

train_features_scaled = scaler.transform(train_df[feature_cols])
val_features_scaled = scaler.transform(val_df[feature_cols])
test_features_scaled = scaler.transform(test_df[feature_cols])


