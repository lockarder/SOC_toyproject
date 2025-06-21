import os
import scipy.io
import numpy as np
import pandas as pd

def cycle_time_to_str(cycle_time_arr):
    """
    将 cycle 时间数组转换为标准时间字符串格式
    """
    return f"{int(cycle_time_arr[0]):04d}-{int(cycle_time_arr[1]):02d}-{int(cycle_time_arr[2]):02d} " \
           f"{int(cycle_time_arr[3]):02d}:{int(cycle_time_arr[4]):02d}:{int(cycle_time_arr[5]):02d}"

def parse_mat_to_df(mat_path):
    """
    解析单个 .mat 文件，返回所有非impedance周期数据的合并 DataFrame（含SOC计算）

    参数：
        mat_path: str，.mat 文件路径

    返回：
        pd.DataFrame，合并的周期数据，含SOC和容量字段
    """
    print(f"Loading file: {mat_path}")
    mat_data = scipy.io.loadmat(mat_path)

    base_name = os.path.splitext(os.path.basename(mat_path))[0]
    print(f"Expected variable name: '{base_name}'")
    available_vars = [k for k in mat_data.keys() if not k.startswith('__')]
    print(f"Variables found in .mat file: {available_vars}")

    if base_name in mat_data:
        battery_data = mat_data[base_name]
        print(f"Using variable '{base_name}' from the file.")
    else:
        battery_key = available_vars[0]
        battery_data = mat_data[battery_key]
        print(f"Warning: Expected variable '{base_name}' not found. Using '{battery_key}' instead.")

    cycles = battery_data['cycle'][0, 0][0]
    all_dfs = []

    for i, cycle in enumerate(cycles):
        cycle_type = cycle['type'][0]
        if cycle_type == 'impedance':
            continue

        data = cycle['data'][0, 0]
        time_arr = data['Time'].flatten()
        voltage = data['Voltage_measured'].flatten()
        current = data['Current_measured'].flatten()
        temperature = data['Temperature_measured'].flatten()
        delta_t = np.diff(time_arr, prepend=time_arr[0])

        # 初始化默认值
        soc = np.full_like(current, np.nan, dtype=np.float64)
        capacity = np.nan

        if cycle_type in ['charge', 'discharge']:
            # 先根据电流方向判断是否反转，正为充电，负为放电
            signed_current = current
            ah_integrated = np.cumsum(signed_current * delta_t) / 3600  # 单位：Ah

            if cycle_type == 'charge':
                capacity = ah_integrated.max() if len(ah_integrated) > 0 else np.nan
                if capacity > 0:
                    soc = ah_integrated / capacity
                else:
                    soc = np.zeros_like(ah_integrated)

            elif cycle_type == 'discharge':
                if 'Capacity' in data.dtype.names:
                    capacity = data['Capacity'][0][0]
                if capacity > 0:
                    soc = 1 + ah_integrated / capacity
                else:
                    soc = np.ones_like(ah_integrated)

        # 时间格式字符串
        cycle_time_arr = cycle['time'][0]
        cycle_start_time_str = cycle_time_to_str(cycle_time_arr)

        df = pd.DataFrame({
            'Cycle_Relative_Time': time_arr,
            'Delta_t': delta_t,
            'Voltage': voltage,
            'Current': current,
            'Temperature': temperature,
            'Cycle_Index': i,
            'Cycle_Type': cycle_type,
            'Cycle_StartTime': cycle_start_time_str,
            'Capacity': capacity,
            'SOC': soc
        })
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df

def batch_convert_mat_to_csv(mat_dir, csv_dir, mat_files=None):
    os.makedirs(csv_dir, exist_ok=True)
    if mat_files is None:
        mat_files = sorted([f for f in os.listdir(mat_dir) if f.lower().endswith('.mat')])
    for mat_file in mat_files:
        mat_path = os.path.join(mat_dir, mat_file)
        csv_filename = os.path.splitext(mat_file)[0] + '_soc.csv'
        csv_path = os.path.join(csv_dir, csv_filename)
        print("=" * 60)
        print(f"🔍 Processing file: {mat_file}")
        print(f"➡️  Full path: {mat_path}")
        print(f"📄 Output CSV: {csv_path}")
        try:
            df = parse_mat_to_df(mat_path)
            print(f"\n📊 Data Preview for {mat_file}:")
            print(df.head())
            print(f"\n📈 数据总行数: {len(df)}")
            print(f"🔁 包含cycle数: {df['Cycle_Index'].nunique()}")
            print(f"🧪 含SOC数据样本数: {df['SOC'].notna().sum()}")
            df.to_csv(csv_path, index=False)
            print(f"\n✅ Saved CSV to: {csv_path}\n")
        except Exception as e:
            print(f"❌ Error processing {mat_file}: {e}")
        print("-" * 60)
