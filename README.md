# SOC_toyproject

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

## 环境依赖
请使用以下命令安装依赖：
```bash
pip install -r requirements.txt
