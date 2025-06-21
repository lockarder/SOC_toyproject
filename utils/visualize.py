import scipy.io
import os
import matplotlib.pyplot as plt

def visualize_battery_cycle(data_folder, mat_filename, cycle_index, visualize=True):
    """
    加载指定 .mat 文件，打印 cycle 数量和指定 cycle 的部分信息，
    并可视化该 cycle 的 data 字段随时间变化图。

    参数:
    - data_folder: str, 数据所在文件夹路径
    - mat_filename: str, .mat 文件名
    - cycle_index: int, 要查看的 cycle 索引（从0开始）
    - visualize: bool, 是否绘图，默认 True

    返回:
    - total_cycles: int, 该文件中 cycle 总数量
    """
    mat_path = os.path.join(data_folder, mat_filename)
    mat_data = scipy.io.loadmat(mat_path)

    var_name = os.path.splitext(mat_filename)[0]
    battery_data = mat_data.get(var_name)
    if battery_data is None:
        raise ValueError(f"未找到对应变量 {var_name} 在 {mat_filename} 中")

    battery_struct = battery_data[0, 0]
    cycles = battery_struct['cycle']

    # 支持多维或单维cycle结构
    total_cycles = cycles.shape[1] if len(cycles.shape) > 1 else cycles.shape[0]
    print(f"总循环次数: {total_cycles}")

    # 获取指定cycle，兼容单维和二维情况
    if total_cycles == 1:
        cycle = cycles[0]
    else:
        cycle = cycles[0, cycle_index] if len(cycles.shape) > 1 else cycles[cycle_index]

    print(f"Cycle {cycle_index} 类型:", cycle['type'][0])
    print(f"Cycle {cycle_index} 环境温度:", cycle['ambient_temperature'])
    print(f"Cycle {cycle_index} 时间长度:", cycle['time'].shape)

    if not visualize:
        return total_cycles

    # 访问 data 字段，可能有多层嵌套
    data = cycle['data']
    data = data[0, 0] if len(data.shape) > 1 else data[0]

    time = data['Time'].flatten()

    # 可能有字段不存在，安全访问
    fields_to_plot = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_charge', 'Voltage_charge']
    available_fields = [f for f in fields_to_plot if f in data.dtype.names]

    plt.figure(figsize=(12, 3 * len(available_fields)))
    for i, field in enumerate(available_fields, 1):
        plt.subplot(len(available_fields), 1, i)
        plt.plot(time, data[field].flatten())
        plt.title(f'{field} vs Time (Cycle {cycle_index})')
        plt.xlabel('Time (min)')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

    return total_cycles




