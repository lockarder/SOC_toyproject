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

class Basic1DCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_layers: int = 2,
        stride: int = 1,                 # 新增 stride 参数，默认1（不缩短序列）
        pool_type: str = "avg",          # 'avg' or 'max'
        dropout: float = 0.2
    ):
        super(Basic1DCNN, self).__init__()

        conv_layers = []
        in_channels = input_size

        for i in range(num_layers):
            conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=stride,               # 使用传入的 stride
                padding=kernel_size // 2     # 保持卷积核中心对齐，padding固定
            ))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(num_filters))
            conv_layers.append(nn.Dropout(dropout))
            in_channels = num_filters

        self.conv_stack = nn.Sequential(*conv_layers)

        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}")

        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        # 输入形状 (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)            # → (batch, input_size, seq_len)
        x = self.conv_stack(x)            # → (batch, num_filters, new_seq_len)
        x = self.pool(x)                  # → (batch, num_filters, 1)
        x = x.squeeze(-1)                 # → (batch, num_filters)
        x = self.fc(x)                    # → (batch, 1)
        return x
    
class BasicCNNLSTMParallel(nn.Module):
    def __init__(
        self,
        input_size: int,
        cnn_params: dict,
        lstm_params: dict,
        fc_hidden_size: int = 64,
        dropout: float = 0.3
    ):
        """
        并行 CNN + LSTM 特征融合模型，用于 SOC 估计等时序回归任务

        参数：
            input_size (int): 每个时间步的特征维度
            cnn_params (dict): 包含CNN的超参数（num_filters, kernel_size等）
            lstm_params (dict): 包含LSTM的超参数（hidden_size, num_layers等）
            fc_hidden_size (int): 融合后全连接隐藏层维度
            dropout (float): 融合层 dropout 概率
        """
        super(BasicCNNLSTMParallel, self).__init__()

        # CNN 分支
        self.cnn_branch = Basic1DCNN(
            input_size=input_size,
            num_filters=cnn_params.get("num_filters", 64),
            kernel_size=cnn_params.get("kernel_size", 3),
            num_layers=cnn_params.get("num_layers", 2),
            stride=cnn_params.get("stride", 1),
            pool_type=cnn_params.get("pool_type", "avg"),
            dropout=cnn_params.get("dropout", 0.2)
        )

        # LSTM 分支
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_params.get("hidden_size", 64),
            num_layers=lstm_params.get("num_layers", 2),
            batch_first=True,
            dropout=lstm_params.get("dropout", 0.0) if lstm_params.get("num_layers", 2) > 1 else 0.0
        )
        self.lstm_hidden_size = lstm_params.get("hidden_size", 64)

        # 融合后回归头
        self.regressor = nn.Sequential(
            nn.Linear(cnn_params.get("num_filters", 64) + self.lstm_hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, 1)
        )

    def forward(self, x):
        # 输入 x: (batch, seq_len, input_size)
        x_cnn = x.permute(0, 2, 1)                          # CNN 输入: (batch, input_size, seq_len)
        cnn_feat = self.cnn_branch.conv_stack(x_cnn)        # → (batch, num_filters, new_seq_len)
        cnn_feat = self.cnn_branch.pool(cnn_feat).squeeze(-1)  # → (batch, num_filters)

        # LSTM 分支输出（取最后一层的最后时间步）
        _, (h_n, _) = self.lstm(x)                          # h_n: (num_layers, batch, hidden_size)
        lstm_feat = h_n[-1]                                 # → (batch, hidden_size)

        # 特征融合
        combined = torch.cat([cnn_feat, lstm_feat], dim=1)  # → (batch, num_filters + hidden_size)
        out = self.regressor(combined)                      # → (batch, 1)
        return out
