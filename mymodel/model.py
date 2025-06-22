import torch
import torch.nn as nn

class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        """
        基础 LSTM 模型，用于序列回归任务（例如电池 SOC 估计）

        参数：
            input_size (int): 输入特征数量（如 Voltage, Current, 等）
            hidden_size (int): LSTM 隐状态维度
            num_layers (int): LSTM 层数
            dropout (float): LSTM 层之间的 dropout 概率（num_layers > 1 时有效）
        """
        super(BasicLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_size, 1)  # 输出一个标量（SOC）

    def forward(self, x):
        """
        前向传播
        输入：
            x: shape [batch_size, seq_len, input_size]
        输出：
            y: shape [batch_size, 1]
        """
        out, _ = self.lstm(x)                  # out: [batch, seq_len, hidden_size]
        last_time_step = out[:, -1, :]         # 取最后一个时间步的输出
        y = self.fc(last_time_step)            # 映射为 [batch, 1]
        return y.squeeze(1)                    # 输出 [batch]
