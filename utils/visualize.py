import scipy.io
import os
import matplotlib.pyplot as plt
import numpy as np

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
