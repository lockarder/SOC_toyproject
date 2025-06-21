import sys
import os
import pandas as pd
from utils import plot_voltage_soc_by_cycle
from utils import parse_mat_to_df
from utils import batch_convert_mat_to_csv
from utils import clean_soc_csv_files

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
clean_soc_csv_files("data/processed")







