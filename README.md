# NACA翼型流场预测项目

## 项目简介

本项目基于**图神经网络（GNN）**实现NACA翼型绕流流场的快速预测。通过训练深度学习模型，可以根据翼型几何形状、来流速度(velocity)和攻角(angle of attack)等参数，预测翼型周围的压力场分布。

## 技术栈

- **深度学习框架**: PyTorch + PyTorch Geometric
- **图神经网络**: 支持 GCN、GAT、GraphSAGE 三种架构
- **数据格式**: VTU/VTP (VTK格式) → PKL (Pickle格式)
- **可视化**: Matplotlib, PyVista
- **硬件支持**: CPU / NVIDIA GPU（CUDA）

## 项目结构

```
NACA/
├── train_airfoil.py          # 模型训练主程序
├── test_train_airfoil.py     # 模型测试/推理程序
├── model.py                  # 神经网络模型定义
├── data.py                   # 数据集加载与预处理
├── vtu2pt.py                 # VTU/VTP文件转换为PKL格式
├── case2vtu-cell2point-all_in.py  # CASE文件转VTU格式
├── log.py                    # 日志工具
├── case1/                    # 训练结果保存目录
│   └── model.pkl             # 训练好的模型权重
├── data2/                    # 处理后的数据集目录
│   ├── factors.pkl           # 边特征归一化因子
│   ├── train_max_min.pkl     # 场量归一化参数
│   ├── outputs_train/        # 训练集
│   ├── outputs_val/          # 验证集
│   └── outputs_test/         # 测试集
└── naca_shuiyi_vtu/          # 原始VTU仿真数据
    └── *.vtu                 # 不同工况的流场数据
```

## 模型架构

项目采用 **Encode-Process-Decode** 架构：

```
输入特征 → Encoder(MLP) → GNN Processor → Decoder(MLP) → 输出预测
```

### 输入特征
- **节点特征 (x)**: 节点坐标 (x, y) + 归一化攻角 + 归一化速度
- **边特征 (edge_attr)**: 节点间欧氏距离（归一化后）

### 图神经网络处理器 (可选)
| 模型 | 描述 |
|------|------|
| **GCN** | 图卷积网络 (GCNConv) |
| **GAT** | 图注意力网络 (GATConv) - 默认 |
| **SAGE** | GraphSAGE 采样聚合网络 |

### 输出
- 预测的压力场分布

## 数据格式

### 原始数据命名规则
```
aoa{攻角}_v{速度}_y+30_square.vtu
```
例如: `aoa5_v2_y+30_square.vtu` 表示攻角5°、来流速度2m/s的工况

### 数据处理流程
```
CASE文件 → (case2vtu) → VTU文件 → (vtu2pt) → PKL文件 → 模型训练
```

## 快速开始

### 环境依赖

```bash
pip install torch torch_geometric pyvista numpy matplotlib
# 如需使用CUDA GPU加速
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 1. 数据预处理

将VTU文件转换为训练所需的PKL格式：

```python
# 编辑 vtu2pt.py 中的路径配置
input_dir = 'path/to/vtu/files/'
output_dir = 'path/to/output/'

# 运行转换
python vtu2pt.py
```

### 2. 模型训练

```python
# 编辑 train_airfoil.py 中的配置
program_path = '/your/project/path'
n_epoch = 150
modelname = 'GAT'  # 可选: 'GCN', 'GAT', 'SAGE'

# 开始训练
python train_airfoil.py
```

### 3. 模型测试

```python
# 编辑 test_train_airfoil.py 中的配置
program_path = '/your/project/path'
case_folder = 'case1'  # 模型保存目录

# 运行测试
python test_train_airfoil.py
```

## 主要参数配置

### 模型参数 (train_airfoil.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_channels` | 128 | 隐藏层通道数 |
| `proc_layers` | 6 | GNN处理层数 |
| `latent_size` | 128 | 潜在特征维度 |
| `num_layers` | 2 | MLP层数 |
| `batchsize` | 1 | 批次大小 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_epoch` | 150 | 训练轮数 |
| `learning_rate` | 0.0005 | 学习率 |
| `weight_decay` | 1e-3 | 权重衰减 |
| `gamma` | 0.5 | 学习率衰减因子 |
| `lambda_gnn` | 0.8 | GNN残差连接系数 |

### GAT特定参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `heads` | 2 | 注意力头数 |
| `slope` | 0.2 | LeakyReLU斜率 |
| `dropout` | 0.0 | Dropout比率 |

## 分布式训练

项目支持基于CUDA的多GPU分布式训练：

```python
distributed = True  # 启用分布式
# 配置主节点
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29502"
# 使用NCCL后端
torch.distributed.init_process_group(backend="nccl", ...)
```

## 可视化输出

训练和测试过程中会生成流场可视化图像，包括：
- **真实场 (true)**: CFD仿真结果
- **预测场 (pred)**: 神经网络预测结果
- **差值场 (diff)**: 预测与真实的误差分布

## 文件说明

| 文件 | 功能描述 |
|------|----------|
| `model.py` | 定义Encoder、Decoder、MeshGCN、MeshGAT、MeshSAGE等网络模块 |
| `data.py` | MeshAirfoilDataset数据集类、数据预处理、可视化函数 |
| `train_airfoil.py` | 完整的训练流程，包括数据加载、模型初始化、训练循环、验证 |
| `test_train_airfoil.py` | 加载训练好的模型进行测试和结果可视化 |
| `vtu2pt.py` | 将VTU格式的CFD结果转换为图数据格式 |
| `case2vtu-cell2point-all_in.py` | 将CASE格式转换为VTU格式，并处理单元数据到点数据的转换 |
| `log.py` | 日志记录工具 |

## 数据集

`naca_shuiyi_vtu/` 目录包含多工况的NACA翼型CFD仿真数据：
- **攻角范围**: 0° - 10°
- **速度范围**: 0.25 - 5 m/s
- **网格类型**: 非结构化网格（y+ ~ 30）

## 许可证

本项目仅供学术研究使用。

## 参考文献

- Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks", ICLR 2021
- PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/
