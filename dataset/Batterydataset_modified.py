import torch
from torch.utils.data import Dataset

class BatteryDatasetLSTM(Dataset):
    """
    自定义电池数据集类：用于 LSTM 序列预测模型中，提取滑动窗口样本。
    
    每个样本由序列长度为 sequence_length 的输入组成，预测的是该序列末尾时间点的 SOC 值。
    滑窗严格限定在单个 SourceFile + Cycle_Index 内，确保物理过程连贯。

    参数：
        df (pd.DataFrame): 包含所有特征、SOC、Cycle_Index、SourceFile 的 DataFrame。
        feature_cols (list[str]): 输入特征列名，例如 ['Voltage', 'Current', 'Temperature', 'Delta_t']
        label_col (str): 标签列名，通常为 'SOC'
        sequence_length (int): LSTM 序列长度（滑窗宽度）
    """
    def __init__(self, df, feature_cols, label_col, sequence_length=20):
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.label_col = label_col

        self.samples = []

        # 保证顺序性并按 cycle 分组
        self.data = df.reset_index(drop=True)
        grouped = self.data.groupby(['SourceFile', 'Cycle_Index'])

        for (src_file, cycle_idx), group in grouped:
            if len(group) < sequence_length:
                continue  # 跳过长度不够的周期

            # 转为 float32 的张量（Tensor），供 PyTorch 使用
            features = torch.tensor(group[feature_cols].values, dtype=torch.float32)
            labels = torch.tensor(group[label_col].values, dtype=torch.float32)

            # 滑动窗口，生成所有可用的 (X_seq, y) 对
            for i in range(len(group) - sequence_length + 1):
                x_seq = features[i : i + sequence_length]         # shape: [sequence_length, input_dim]
                y = labels[i + sequence_length - 1]               # 预测序列最后一步的 SOC
                self.samples.append((x_seq, y))                   # 可选：加上 (src_file, cycle_idx) debug 时有用

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # 返回一个 (seq_x, y) 对
