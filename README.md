# OOD-Tier SGCN: Out-of-Distribution Detection for Graph Neural Networks

[中文版本](#中文版本) | [English](#english-version)

---

## English Version

### Overview

**OOD-Tier SGCN** is a comprehensive framework for out-of-distribution (OOD) detection in graph neural networks (GNNs). This repository implements state-of-the-art methods for detecting OOD samples in graph-based learning scenarios, supporting both classification and detection tasks across multiple types of distribution shifts (structure, label, and feature).

### Key Features

- **Multiple OOD Detection Methods**: Implements baseline methods (MSP, ODIN, Mahalanobis) and advanced methods (GNNSafe)
- **Diverse Backbone Networks**: Supports GCN, GAT, SGC, MixHop, H2GCN, GPRGNN, and more
- **Flexible Distribution Shifts**: Handles structure, label, and feature-based OOD scenarios
- **Multi-Dataset Support**: Cora, Citeseer, PubMed, Proteins, PPI, Twitch, Amazon, ArXiv
- **Comprehensive Evaluation**: Metrics include AUROC, AUPR, FPR95, and classification accuracy
- **Energy-Based Methods**: Advanced energy regularization and belief propagation techniques

### Repository Structure

```
├── main.py                          # Main training and evaluation pipeline
├── parse.py                         # Argument parser and configuration
├── backbone.py                      # GNN backbone architectures
├── baselines.py                     # Baseline OOD detection methods
├── gnnsafe.py                       # GNNSafe implementation
│
├── dataset.py                       # Dataset loading utilities
├── data_utils.py                    # Data preprocessing and evaluation utilities
├── dataset.py                       # Dataset-specific implementations
│
├── logger.py                        # Logging and result tracking
├── ogb_compat.py                    # OGB compatibility utilities
│
├── two_stage_mixed_test.py          # Two-stage OOD detection on mixed test sets
├── two_stage_utils.py               # Utility functions for two-stage methods
├── two_stage_arxiv.py               # ArXiv-specific two-stage implementation
│
├── discuss.py                       # Analysis and discussion utilities
├── gnnsafe.py                       # Additional GNNSafe components
├── diagnose_arxiv_filtering.py      # ArXiv filtering diagnostics
│
├── run.sh                           # Main training script
├── run_baseline.sh                  # Baseline method training
├── run_discuss.sh                   # Discussion experiments
├── run_hyper_search.sh              # Hyperparameter search
├── run_pubmed.sh                    # PubMed dataset training
├── run_visualize.sh                 # Visualization script
├── run_remote_mixed_test_amazon.sh  # Amazon remote testing
├── run_remote_mixed_test_amazon_matrix.sh  # Amazon matrix testing
│
└── LICENSE                          # MIT License
```

### Core Modules

#### 1. **Backbone Networks** (`backbone.py`)
- **MLP**: Multi-layer perceptron with batch normalization
- **SGC**: Simplified Graph Convolution
- **GCN**: Graph Convolutional Network with batch normalization
- **GAT**: Graph Attention Network
- **MixHop**: Multi-hop aggregation network
- **H2GCN**: Second-order Homophily-preserving GCN
- **APPNP**: Approximate Personalized PageRank
- **GPRGNN**: Generalized PageRank Neural Network

#### 2. **OOD Detection Methods** (`baselines.py`, `gnnsafe.py`)
- **MSP** (Maximum Softmax Probability): Baseline confidence-based detection
- **ODIN**: Temperature scaling and input perturbation
- **Mahalanobis**: Distance-based detection using Mahalanobis distance
- **OE** (Outlier Exposure): OOD detection via exposure to auxiliary data
- **GNNSafe**: Advanced energy-based method with belief propagation

#### 3. **Data Utilities** (`data_utils.py`)
- Graph normalization and adjacency matrix operations
- Train/validation/test split management
- Evaluation metrics: accuracy, ROC-AUC, F1 score
- GPU memory monitoring

#### 4. **Dataset Management** (`dataset.py`)
- Support for benchmark datasets (Cora, Citeseer, PubMed)
- Support for large-scale datasets (Proteins, PPI, Twitch)
- OOD train/test set generation
- Flexible split strategies

### Usage

#### Basic Training
```bash
python main.py --dataset cora --method gnnsafe --backbone gcn --epochs 200 --mode detect
```

#### Hyperparameter Tuning
```bash
bash run_hyper_search.sh
```

#### Baseline Comparison
```bash
bash run_baseline.sh
```

#### PubMed Dataset
```bash
bash run_pubmed.sh
```

#### Visualization
```bash
bash run_visualize.sh
```

### Configuration Parameters

Key arguments in `parse.py`:

- `--dataset`: Dataset name (cora, citeseer, pubmed, proteins, ppi, twitch, amazon, arxiv)
- `--ood_type`: Type of OOD shift (structure, label, feature)
- `--method`: Detection method (msp, gnnsafe, odin, mahalanobis, oe)
- `--backbone`: GNN architecture (gcn, gat, sgc, mixhop, h2gcn, appnp, gprgnn)
- `--mode`: Task mode (classify, detect)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--hidden_channels`: Hidden dimension size
- `--T`: Temperature for softmax scaling
- `--use_reg`: Enable energy regularization
- `--use_prop`: Enable energy belief propagation

### Evaluation Metrics

- **Classification Mode**:
  - Accuracy on train/val/test splits
  
- **Detection Mode**:
  - AUROC (Area Under ROC Curve)
  - AUPR (Area Under Precision-Recall Curve)
  - FPR95 (False Positive Rate at 95% TPR)
  - Test Score

### Advanced Features

#### Energy-Based OOD Detection
- Energy regularization loss for better calibrated energy scores
- Energy belief propagation for leveraging graph structure
- Configurable energy aggregation strategies (mean, median, trimmed-mean)

#### Subgraph-level OOD Detection (SGCN)
- Subgraph Convolutional Network integration
- Energy-driven subgraph reweighting
- Multiple subgraph energy aggregation methods

---

## 中文版本

### 概述

**OOD-Tier SGCN** 是一个针对图神经网络 (GNN) 分布外 (OOD) 检测的综合框架。该仓库实现了最先进的 OOD 检测方法，支持多种分布偏移类型（结构、标签、特征），适用于分类和检测任务。

### 主要特性

- **多种 OOD 检测方法**：实现基础方法（MSP、ODIN、Mahalanobis）和高级方法（GNNSafe）
- **多样化的骨干网络**：支持 GCN、GAT、SGC、MixHop、H2GCN、GPRGNN 等
- **灵活的分布偏移**：处理基于结构、标签和特征的 OOD 场景
- **多数据集支持**：Cora、Citeseer、PubMed、Proteins、PPI、Twitch、Amazon、ArXiv
- **综合评估指标**：AUROC、AUPR、FPR95 和分类准确率
- **基于能量的方法**：先进的能量正则化和置信度传播技术

### 仓库结构

```
├── main.py                          # 主训练和评估管道
├── parse.py                         # 参数解析和配置
├── backbone.py                      # GNN 骨干网络架构
├── baselines.py                     # 基础 OOD 检测方法
├── gnnsafe.py                       # GNNSafe 实现
│
├── dataset.py                       # 数据集加载工具
├── data_utils.py                    # 数据预处理和评估工具
├── dataset.py                       # 数据集特定实现
│
├── logger.py                        # 日志记录和结果追踪
├── ogb_compat.py                    # OGB 兼容性工具
│
├── two_stage_mixed_test.py          # 混合测试集上的两阶段 OOD 检测
├── two_stage_utils.py               # 两阶段方法的工具函数
├── two_stage_arxiv.py               # ArXiv 特定的两阶段实现
│
├── discuss.py                       # 分析和讨论工具
├── gnnsafe.py                       # 额外的 GNNSafe 组件
├── diagnose_arxiv_filtering.py      # ArXiv 过滤诊断
│
├── run.sh                           # 主训练脚本
├── run_baseline.sh                  # 基础方法训练
├── run_discuss.sh                   # 讨论实验
├── run_hyper_search.sh              # 超参数搜索
├── run_pubmed.sh                    # PubMed 数据集训练
├── run_visualize.sh                 # 可视化脚本
├── run_remote_mixed_test_amazon.sh  # Amazon 远程测试
├── run_remote_mixed_test_amazon_matrix.sh  # Amazon 矩阵测试
│
└── LICENSE                          # MIT 许可证
```

### 核心模块

#### 1. **骨干网络** (`backbone.py`)
- **MLP**：具有批量归一化的多层感知器
- **SGC**：简化图卷积
- **GCN**：具有批量归一化的图卷积网络
- **GAT**：图注意力网络
- **MixHop**：多跳聚合网络
- **H2GCN**：二阶同质性保留 GCN
- **APPNP**：近似个性化 PageRank
- **GPRGNN**：广义 PageRank 神经网络

#### 2. **OOD 检测方法** (`baselines.py`, `gnnsafe.py`)
- **MSP**（最大 Softmax 概率）：基于置信度的检测基线
- **ODIN**：温度缩放和输入扰动
- **Mahalanobis**：基于 Mahalanobis 距离的检测
- **OE**（异常值暴露）：通过辅助数据进行 OOD 检测
- **GNNSafe**：具有置信度传播的先进能量方法

#### 3. **数据工具** (`data_utils.py`)
- 图规范化和邻接矩阵操作
- 训练/验证/测试分割管理
- 评估指标：准确率、ROC-AUC、F1 分数
- GPU 内存监控

#### 4. **数据集管理** (`dataset.py`)
- 支持基准数据集（Cora、Citeseer、PubMed）
- 支持大规模数据集（Proteins、PPI、Twitch）
- OOD 训练/测试集生成
- 灵活的分割策略

### 使用方法

#### 基础训练
```bash
python main.py --dataset cora --method gnnsafe --backbone gcn --epochs 200 --mode detect
```

#### 超参数调整
```bash
bash run_hyper_search.sh
```

#### 基础方法对比
```bash
bash run_baseline.sh
```

#### PubMed 数据集
```bash
bash run_pubmed.sh
```

#### 可视化
```bash
bash run_visualize.sh
```

### 配置参数

`parse.py` 中的关键参数：

- `--dataset`：数据集名称（cora、citeseer、pubmed、proteins、ppi、twitch、amazon、arxiv）
- `--ood_type`：OOD 偏移类型（structure、label、feature）
- `--method`：检测方法（msp、gnnsafe、odin、mahalanobis、oe）
- `--backbone`：GNN 架构（gcn、gat、sgc、mixhop、h2gcn、appnp、gprgnn）
- `--mode`：任务模式（classify、detect）
- `--epochs`：训练周期数
- `--lr`：学习率
- `--hidden_channels`：隐藏维度大小
- `--T`：Softmax 温度
- `--use_reg`：启用能量正则化
- `--use_prop`：启用能量置信度传播

### 评估指标

- **分类模式**：
  - 训练/验证/测试分割上的准确率
  
- **检测模式**：
  - AUROC（ROC 曲线下面积）
  - AUPR（精准召回曲线下面积）
  - FPR95（95% TPR 时的假正例率）
  - 测试分数

### 高级特性

#### 基于能量的 OOD 检测
- 能量正则化损失，用于更好的能量分数校准
- 能量置信度传播，利用图结构
- 可配置的能量聚合策略（平均值、中位数、截断平均值）

#### 子图级 OOD 检测 (SGCN)
- 子图卷积网络集成
- 能量驱动的子图重加权
- 多种子图能量聚合方法

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@repository{ood-tier-sgcn,
  title={OOD-Tier SGCN: Out-of-Distribution Detection for Graph Neural Networks},
  author={Gwensa0724},
  year={2026},
  url={https://github.com/Gwensa0724/ood-tier_SGCN}
}
```
