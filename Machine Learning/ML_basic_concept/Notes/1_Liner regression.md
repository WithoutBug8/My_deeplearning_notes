# 线性回归（Linear Regression）学习笔记

[返回顶部](#线性回归linear-regression学习笔记)

## 目录

- [1. 线性回归方程](#1-线性回归方程)
  - [1.1 单特征模型](#11-单特征模型)
  - [1.2 多特征模型](#12-多特征模型)
- [2. 损失函数](#2-损失函数)
  - [2.1 损失的距离概念](#21-损失的距离概念)
  - [2.2 常见损失类型对比](#22-常见损失类型对比)
  - [2.3 如何选择损失函数](#23-如何选择损失函数)
- [3. 梯度下降法](#3-梯度下降法)
  - [3.1 梯度下降基本流程](#31-梯度下降基本流程)
  - [3.2 模型收敛与凸函数](#32-模型收敛与凸函数)
- [4. 超参数](#4-超参数)
  - [4.1 学习率（Learning Rate）](#41-学习率learning-rate)
  - [4.2 批次大小（Batch Size）](#42-批次大小batch-size)
  - [4.3 Epoch（训练轮次）](#43-epoch训练轮次)
  - [4.4 用批改作业类比理解 Batch Size & Epoch](#44-用批改作业类比理解-batch-size--epoch)

---

## 1. 线性回归方程

线性回归的目标是用一条（或一个超平面）直线来拟合数据，预测连续值。

### 1.1 单特征模型

基本形式：

$$
\hat{y} = b + w_1 x_1
$$

- $\hat{y}$：预测值（predicted label）
- $b$：**偏置（bias）**，相当于直线的 y 截距（有时记作 $w_0$）
- $w_1$：**权重（weight）**，相当于斜率
- $x_1$：输入特征（feature）

训练目标：通过数据学习最优的 $b$ 和 $w$，使预测尽可能接近真实值。

### 1.2 多特征模型

实际问题通常涉及多个特征：

$$
\hat{y} = b + w_1 x_1 + w_2 x_2 + w_3 x_3 + \dots + w_n x_n
$$

示例特征（预测汽车价格）：

- 发动机排量
- 马力
- 汽缸数量
- 加速能力
- 重量

![五特征线性回归方程示意图](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/equation-multiple-features.png)

{#多特征模型图}

---

## 2. 损失函数

损失（Loss）量化**预测值与真实值之间的差距**。训练目标：最小化损失。

![损失线示意图（箭头表示预测与真实的差距）](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/loss-lines.png)

### 2.1 损失的距离概念

损失只关心**距离大小**，不关心方向（正/负），常用方法：

- 绝对值（去符号）
- 平方（放大/缩小误差）

### 2.2 常见损失类型对比

| 损失类型          | 定义                               | 公式                                      | 特点                                   |
|-------------------|------------------------------------|-------------------------------------------|----------------------------------------|
| L1 损失           | 绝对值误差总和                     | $\sum |y - \hat{y}|$                      | 对离群值不敏感                         |
| MAE（平均绝对误差）| L1 的平均值                        | $\frac{1}{N} \sum |y - \hat{y}|$          | 直观，与标签单位相同                   |
| L2 损失           | 平方误差总和                       | $\sum (y - \hat{y})^2$                    | 对大误差惩罚更重（平方放大）           |
| MSE（均方误差）    | L2 的平均值                        | $\frac{1}{N} \sum (y - \hat{y})^2$        | 最常用，优化平滑                       |
| RMSE              | MSE 的平方根                       | $\sqrt{\frac{1}{N} \sum (y - \hat{y})^2}$ | 单位与标签相同，更易解释               |

MSE vs MAE 对离群值的影响：

![MSE 模型更靠近离群值](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/model-mse.png)

![MAE 模型远离离群值](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/model-mae.png)

### 2.3 如何选择损失函数

- **选 MSE**（或 L2）：
  - 希望对**大误差**严厉惩罚
  - 离群值可能包含重要信息
  - 优化更平滑

- **选 MAE**（或 L1）：
  - 数据有明显离群值，不希望过度影响
  - 想要更鲁棒（robust）的模型
  - 损失更直观（平均误差大小）

业务角度：哪种误差代价更高？

---

## 3. 梯度下降法

梯度下降（Gradient Descent）是优化权重和偏置的核心算法，通过迭代逐步减小损失。

### 3.1 梯度下降基本流程

1. 随机初始化权重和偏置（接近 0）
2. 用当前参数计算预测 → 计算损失
3. 计算梯度（损失对每个参数的偏导数）
4. 沿**负梯度方向**更新参数：  
   新参数 = 旧参数 - 学习率 × 梯度
5. 重复 2–4，直到损失不再显著下降（收敛）

### 3.2 模型收敛与凸函数

线性回归的损失函数是**凸函数**（convex），只有一个全局最低点 → 保证梯度下降能找到最优解（或非常接近）。

典型损失曲线（收敛过程）：

前几次迭代损失陡降 → 逐渐平缓 → 趋于稳定

![损失曲面（凸形状，梯度下降路径）](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/loss-surface-points.png)

---

## 4. 超参数

超参数（Hyperparameters）由人工设定，控制训练过程；  
参数（Parameters，如 w、b）由模型自动学习。

### 4.1 学习率（Learning Rate）

决定每步参数更新的步长。

- 太小 → 收敛极慢
- 太大 → 震荡 / 发散（损失上升）

![合适学习率：快速收敛](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/correct-lr.png)
合适学习率：快速收敛

![学习率过小：缓慢收敛](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/small-lr.png)
学习率过小：缓慢收敛

![学习率过大：震荡或发散](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/high-lr.png)
学习率过大：震荡或发散

### 4.2 批次大小（Batch Size）

每次更新参数前使用的样本数量。

| 类型                  | Batch Size     | 特点                               | 损失曲线特点     |
|-----------------------|----------------|------------------------------------|------------------|
| 全批量（Full Batch）   | = 全部样本 N   | 梯度精确，但计算慢                 | 非常平滑         |
| 随机梯度下降（SGD）    | = 1            | 噪声大，更新频繁                   | 剧烈波动         |
| 小批量（Mini-batch）   | 1 < size < N   | 折中：噪声适中，计算效率高         | 较平滑，有小波动 |

![Mini-batch SGD 损失曲线（比纯 SGD 平滑）](https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/mini-batch-sgd.png)

适量噪声有时有益（尤其在深度学习中帮助泛化）。

### 4.3 Epoch（训练轮次）

一个 Epoch = 模型完整遍历一次整个训练数据集。

例：  
训练集 10000 样本，batch size = 100 → 1 Epoch = 100 次迭代  
通常训练 5–100+ Epoch，视收敛情况调整。

### 4.4 用批改作业类比理解 Batch Size & Epoch

老师批改 10000 份作业：

- batch size = 100 → 每次批改 100 份
- 1 Epoch = 批改完整 10000 份（100 次批改）
- epoch = 5 → 总共批改 500 次（参数更新 500 次）

---
