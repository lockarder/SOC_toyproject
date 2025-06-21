import os
import pandas as pd

def clean_soc_csv_files(input_dir, output_dir=None, soc_min=0.0, soc_max=1.0):
    """
    批量清洗SOC CSV文件：
      - 删除含NaN的行
      - 保留SOC在指定范围内的行
      - 保存为新文件，文件名加后缀 _clean.csv

    参数：
    - input_dir: str，输入CSV文件目录，默认读取所有 *_soc.csv 文件
    - output_dir: str，输出目录，默认与输入目录相同
    - soc_min: float，SOC下限，默认0.0
    - soc_max: float，SOC上限，默认1.0
    """
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith("_soc.csv"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace("_soc.csv", "_soc_clean.csv"))

            print(f"📂 读取中: {filename}")
            df = pd.read_csv(input_path)

            if 'SOC' not in df.columns:
                print(f"⚠️  跳过：{filename} 不包含 'SOC' 列")
                continue

            original_len = len(df)

            # 删除含NaN的行
            df_clean = df.dropna()

            # 保留SOC范围内的行
            df_clean = df_clean[(df_clean['SOC'] >= soc_min) & (df_clean['SOC'] <= soc_max)]

            print(f"✅ 清洗完成: {filename}")
            print(f"    原始数据行数: {original_len}")
            print(f"    清洗后行数: {len(df_clean)}")
            print(f"    删除行数: {original_len - len(df_clean)}")

            # 保存
            df_clean.to_csv(output_path, index=False)
            print(f"📁 保存到: {output_path}\n")


import os
import pandas as pd

def load_all_clean_csvs(processed_dir, selected_features=None):
    """
    加载指定目录下所有 *_soc_clean.csv 文件，提取指定特征列（含 Cycle_Index），合并成一个 DataFrame。

    参数：
        processed_dir (str): 清洗后 CSV 文件的路径，如 data/processed
        selected_features (list[str] or None): 需要提取的特征列（默认提取 ['Delta_t', 'Voltage', 'Current', 'Temperature', 'SOC']）

    返回：
        pd.DataFrame: 合并后的清洗数据，附带 SourceFile 和 Cycle_Index 列
    """
    if selected_features is None:
        selected_features = ['Delta_t', 'Voltage', 'Current', 'Temperature', 'SOC']

    # 确保 Cycle_Index 一定保留
    if 'Cycle_Index' not in selected_features:
        selected_features = selected_features + ['Cycle_Index']

    all_frames = []

    for filename in os.listdir(processed_dir):
        if filename.endswith('_soc_clean.csv'):
            file_path = os.path.join(processed_dir, filename)
            df = pd.read_csv(file_path)

            # 检查列完整性
            missing_cols = [col for col in selected_features if col not in df.columns]
            if missing_cols:
                raise ValueError(f"⚠️ 文件 {filename} 缺失列: {missing_cols}")
            
            df = df[selected_features].copy()
            df['SourceFile'] = filename
            all_frames.append(df)

    if not all_frames:
        raise FileNotFoundError(f"❌ 未找到任何 '_soc_clean.csv' 文件于目录: {processed_dir}")
    
    combined_df = pd.concat(all_frames, ignore_index=True)
    return combined_df
