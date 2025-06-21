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
