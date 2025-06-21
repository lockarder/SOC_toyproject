# SOC_toyproject
Author: Shiang Guo
## 项目简介
这是一个电池状态（SOC）估计的深度学习示范项目，包含数据处理、模型训练和评估代码。

## 项目结构
- `data/`：存放数据文件
- `notebooks/`：交互式分析笔记本
- `src/`：源代码
- `.gitignore`：Git忽略文件
- `requirements.txt`：依赖库列表

## 数据来源 
NASA 电池数据集（Battery Data Set）
该数据集包含锂离子电池在不同温度下的充放电实验数据，记录了阻抗作为损伤标准。

下载链接：https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

数据集引用：Brad Bebout, Leslie Profert-Bebout, Erich Fleming, Angela Detweiler, and Kai Goebel “Battery Data Set”, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA 
scirp.org
+2
nasa.gov
+2
paperswithcode.com
+2

典型的充放电曲线：
1. S型放电：
![放电](image/放电.png)
2. CC-CV充电
![充电](image/CCCV充电.png)
3. 锂电池放电完成后极化电压消除，电压回升 温度回降
![放电恢复](image/锂电池放电完成后极化电压消除，电压回升%20温度回降.png)

## __main__.py 文件架构 
        Step 1: 原始数据加载
                ↓
        Step 2: 按文件划分训练 / 验证 / 测试
                ↓
        Step 3: 用训练集 fit 标准化器 StandardScaler
                ↓
        Step 4: 分别 transform 三个子集
                ↓
        Step 5: 转为 Tensor
                ↓
        Step 6: 封装成 Dataset（带序列窗口）
                ↓
        Step 7: 封装成 DataLoader（可训练）
1. batch_convert_mat_to_csv（）实现data\raw\ .mat 到 .csv的转换，假如Delta time 和 SOC 的伪真值
2. clean_soc_csv_file （） 清洗NAN和去掉SOC不合理的值 得到data\procssed\ **soc_clean.csv
3. plot_voltage_soc_by_cycle() 可视化观察不同工况的 [V I SOC] with respect to time 
4. df = load_all_clean_csvs(processed_dir) 拼接文件 B0005,6,7 _soc_clean.csv
5. 



## 环境依赖
请使用以下命令安装依赖：
```bash
pip install -r requirements.txt


