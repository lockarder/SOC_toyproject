import scipy.io
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_battery_cycle(data_folder, mat_filename, cycle_index, visualize=True):
    mat_path = os.path.join(data_folder, mat_filename)
    mat_data = scipy.io.loadmat(mat_path)

    battery_data = mat_data.get(os.path.splitext(mat_filename)[0])
    if battery_data is None:
        raise ValueError(f"未找到对应变量 {os.path.splitext(mat_filename)[0]} 在 {mat_filename} 中")

    battery_struct = battery_data[0, 0]
    cycles = battery_struct['cycle']
    total_cycles = cycles.shape[1] if len(cycles.shape) > 1 else cycles.shape[0]

    print(f"总循环次数: {total_cycles}")

    if total_cycles == 1:
        cycle = cycles[0]
    else:
        cycle = cycles[0, cycle_index] if len(cycles.shape) > 1 else cycles[cycle_index]

    print(f"Cycle {cycle_index} 类型:", cycle['type'][0])
    
    # 环境温度格式调整，避免[[24]]这种多维嵌套
    ambient_temp = cycle['ambient_temperature']
    if isinstance(ambient_temp, np.ndarray):
        ambient_temp = ambient_temp.flatten()
        if ambient_temp.size == 1:
            ambient_temp = ambient_temp[0]
    print(f"Cycle {cycle_index} 环境温度:", ambient_temp)

    # 时间长度，展示维度和内容
    print(f"Cycle {cycle_index} time 形状:", cycle['time'].shape)
    print(f"Cycle {cycle_index} time 内容:", cycle['time'].flatten())

    # 读取 data
    data = cycle['data']

    data = data[0, 0] if len(data.shape) > 1 else data[0]
    print(f"Cycle {cycle_index} data time 内容:", data['time'].flatten())
    
    # 提取6个字段
    V = data['Voltage_measured'].flatten()
    I = data['Current_measured'].flatten()
    T = data['Temperature_measured'].flatten()
    I_chg = data['Current_charge'].flatten()
    V_chg = data['Voltage_charge'].flatten()
    time = data['Time'].flatten()

    # 打印前10条数据（保证一定打印）
    combined = np.stack([V, I, T, I_chg, V_chg, time], axis=1)
    print(f"\nCycle {cycle_index} 数据前 10 行 (Voltage, Current, Temperature, Current_charge, Voltage_charge, Time):")
    print(combined[:10])

    # 是否绘图
    if visualize:
        fields_to_plot = [
            ('Voltage_measured', V),
            ('Current_measured', I),
            ('Temperature_measured', T),
            ('Current_charge', I_chg),
            ('Voltage_charge', V_chg)
        ]

        plt.figure(figsize=(12, 10))
        for i, (label, series) in enumerate(fields_to_plot, 1):
            plt.subplot(len(fields_to_plot), 1, i)
            plt.plot(time, series)
            plt.title(f'{label} vs Time (Cycle {cycle_index})')
            plt.xlabel('Time (min)')
            plt.grid(True)
        plt.tight_layout()
        plt.show()
    print(data['Time'].flatten())
    return total_cycles


def plot_voltage_soc_by_cycle(csv_path):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Cycle_Index', 'Cycle_Type', 'Cycle_Relative_Time', 'Voltage', 'SOC'])
    df['Cycle_Index'] = df['Cycle_Index'].astype(int)

    fig1, axs1 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig1.suptitle('Voltage vs Relative Time')

    fig2, axs2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig2.suptitle('SOC vs Relative Time')

    cycle_types = ['charge', 'discharge']

    for i, cycle_type in enumerate(cycle_types):
        subset = df[df['Cycle_Type'] == cycle_type]
        if subset.empty:
            print(f"⚠️ No data for {cycle_type} cycles, skip plotting.")
            continue

        ax_v = axs1[i]
        ax_s = axs2[i]

        for cycle_idx, group in subset.groupby('Cycle_Index'):
            ax_v.plot(group['Cycle_Relative_Time'], group['Voltage'], alpha=0.7)
            ax_s.plot(group['Cycle_Relative_Time'], group['SOC'], alpha=0.7)

        ax_v.set_title(f'{cycle_type.capitalize()} Cycles')
        ax_v.set_ylabel('Voltage (V)')
        ax_v.grid(True)

        ax_s.set_title(f'{cycle_type.capitalize()} Cycles')
        ax_s.set_xlabel('Relative Time (s)')
        ax_s.set_ylabel('SOC')
        ax_s.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


import matplotlib.pyplot as plt

def plot_variable_curve(df, variable, source_file=None, cycle_index=None):
    """
    绘制指定变量随累计时间变化的曲线。
    
    参数：
    - df: DataFrame，包含数据，必须有 'Delta_t' 列
    - variable: str，要绘制的变量名
    - source_file: str，筛选指定文件（可选）
    - cycle_index: int，筛选指定周期（可选）
    """
    plot_df = df.copy()
    if source_file is not None:
        plot_df = plot_df[plot_df['SourceFile'] == source_file]
    if cycle_index is not None:
        plot_df = plot_df[plot_df['Cycle_Index'] == cycle_index]

    if plot_df.empty:
        print("指定条件下无数据，无法绘图")
        return

    # 计算累计时间
    plot_df = plot_df.sort_values('Delta_t').copy()
    plot_df['CumTime'] = plot_df['Delta_t'].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(plot_df['CumTime'], plot_df[variable], marker='.', linestyle='-')
    plt.title(f"{variable} 随累计时间变化曲线\n文件: {source_file if source_file else '全部'} 周期: {cycle_index if cycle_index is not None else '全部'}")
    plt.xlabel('累计时间 (s)')
    plt.ylabel(variable)
    plt.grid(True)
    plt.show()

