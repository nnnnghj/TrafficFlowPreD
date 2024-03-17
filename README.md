# TrafficFlowPreD

本次设计会运用到三种流量预测算法（SAEs、LSTM、GRU）和openai的api连接的微信机器人。

## 1、设计前准备

### 1.1外部环境

由于提高深度学习的稳定性，使用较老版本的环境进行运行。

| 外部           | 版本  |
| -------------- | ----- |
| python         | 3.6   |
| tensorflow-gpu | 1.5.0 |
| keras          | 2.1.3 |
| scikit-learen  | 0.19  |

使用的是miniconda来进行环境配置，先进行python配置。

```shell
conda create -n my-env python=3.6
```

虚拟环境下载结束之后进行环境激活。

```shell
conda activate my-env
```

安装指定版本的库。

```shell
pip install tensorflow-gpu==1.15 keras==2.2.4 scikit-learn==0.22
```

验证安装

```shell
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import keras; print(keras.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
```

这些命令应分别输出`1.15.0`、`2.2.4`和`0.22`，表示已成功安装了所需版本的库。

<!--若是提示没有这几个版本，可能意味着你需要更新你的pip版本-->

```shell
pip install --upgrade pip
```

之后需要将ipykernel安装进虚拟环境中。

```shell
conda install -n my-env ipykernel
ipython kernel install --user --name=my-env --display-name "Python (my-env)"
```

### 1.2数据准备

#### 1.2.1数据选择

首先考虑到项目需要简单、快速开发，就只以单一车道的流量为切入点进行研究。

搜索符合数据特点，且数据量不错的以PeMS数据为项目数据源，数据是从遍布加州所有主要都市区的高速公路系统中的近 40,000 个独立探测器实时收集的。以项目五分钟为集成点进行分析。

#### 1.2.2数据预处理

为了满足项目的数据要求，对项目进行python处理。

数据处理器：datagen.py

```python

```

