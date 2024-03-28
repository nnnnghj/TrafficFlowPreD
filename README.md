# TrafficFlowPreD

本次设计会运用到三种流量预测算法（SAEs、LSTM、GRU）和openai的api连接的微信机器人。

## 1 设计前准备

### 1.1 外部环境

由于提高深度学习的稳定性，使用较老版本的环境进行运行。并使用AutoDL平台进行实验。

<!--AutoDL可以使用特定镜像来进行以下配置-->

| 外部          | 版本             |
| ------------- | ---------------- |
| python        | 3.8(ubuntu20.04) |
| tensorflow    | 2.9.0            |
| scikit-learen | 1.3.2            |

使用的是TensorFolw来进行环境配置，安装指定版本的库。

```shell
pip install scikit-learn==1.3.2
```

验证安装

```shell
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
```

这些命令应分别输出`1.3.2`和`1.3.2`，表示已成功安装了所需版本的库。

<!--若是提示没有这几个版本，可能意味着你需要更新你的pip版本-->

```shell
pip install --upgrade pip
```



### 1.2 数据准备

#### 1.2.1 数据选择

首先考虑到项目需要简单、快速开发，就只以单一车道的流量为切入点进行研究。

搜索符合数据特点，且数据量不错的以PeMS数据为项目数据源，数据是从遍布加州所有主要都市区的高速公路系统中的近 40,000 个独立探测器实时收集的。以项目五分钟为集成点进行分析。

##### 1.2.1.1 PeMS数据来源页面

<!--数据来源页面为 https://pems.dot.ca.gov/ -->

![](https://s2.loli.net/2024/03/28/9Ow1TX2dZBnvpR7.png)

##### 1.2.1.2 注册并登录

![image-20240328201935210](https://s2.loli.net/2024/03/28/Pdiynjltgq5Zaxp.png)

##### 1.2.1.3 选择数据

![image-20240328202213893](https://s2.loli.net/2024/03/28/dcNXPVbT42eZOIp.png)

上述的District可随意选择。

#### 1.2.2 数据预处理

为了满足项目的数据要求，对项目进行python处理。

数据处理器：datagen.py

```python
# 导入pandas库用于数据处理，以及datetime和timedelta用于处理日期和时间
import pandas as pd
from datetime import datetime, timedelta

# 定义新的列名，这些列名将用于后续处理的CSV文件数据
new_headers = [
    'Timestamp', 'Station', 'District', 'Freeway', 'Direction',
    'Lane Type', 'Station Length', 'Samples', '% Observed',
    'Total Flow', 'Avg Occupancy', 'Avg Speed',
    'Lane 1 Samples', 'Lane 1 Flow', 'Lane 1 Avg Occ',
    'Lane 1 Avg Speed', 'Lane 1 Observed'
]

# 设定文件前缀和日期范围，这将用于构建文件路径和文件名
file_prefix = 'd12_text_station_5min_'
start_date = datetime(2024, 1, 1)  # 开始日期
end_date = datetime(2024, 2, 1)  # 结束日期

# 创建一个空的DataFrame对象，用于累积所有处理过的数据
accumulated_data = pd.DataFrame()

# 生成一个包含所需日期范围内所有日期的列表
date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

# 遍历日期列表，对每一个日期进行数据处理
for date in date_list:
    file_suffix = date.strftime('%Y_%m_%d')  # 根据日期生成文件名后缀
    file_path = f"{file_prefix}{file_suffix}.txt.gz"  # 构建完整的文件路径
    # 尝试执行以下代码块
    try:
        # 使用pandas读取gzip压缩的CSV文件
        daily_data = pd.read_csv(
            file_path,
            compression='gzip',  # 指定文件压缩格式
            header=None,  # 指定文件没有表头
            names=new_headers,  # 使用预定义的列名
            usecols=range(len(new_headers))  # 读取所有定义的列
        )
        # 仅保留'Station'列为1223083的数据行
        filtered_data = daily_data[daily_data['Station'] == 1223083].copy()
        # 将过滤后的数据累加到之前创建的空DataFrame中
        accumulated_data = pd.concat([accumulated_data, filtered_data], ignore_index=True)
    # 如果文件未找到，打印文件路径
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    # 捕获并打印其他异常
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# 选择并重命名原始数据集中的部分列，为后续分析做准备
filtered_data = accumulated_data[['Timestamp', 'Lane 1 Flow', 'Lane 1 Observed', '% Observed']].copy()
filtered_data.rename(columns={
    'Timestamp': '5 Minutes',
    'Lane 1 Flow': 'Lane 1 Flow (Veh/5 Minutes)',
    'Lane 1 Observed': '# Lane Points'
}, inplace=True)
# 将'Timestamp'列的类型转换为datetime，并格式化为指定的字符串格式
filtered_data['5 Minutes'] = pd.to_datetime(filtered_data['5 Minutes'])
filtered_data['5 Minutes'] = filtered_data['5 Minutes'].dt.strftime('%m/%d/%Y %H:%M')

# 将数据集分割为训练集和测试集（80%训练集，20%测试集）
split_point = int(len(filtered_data) * 0.8)
train_data = filtered_data[:split_point]
test_data = filtered_data[split_point:]

# 将训练集和测试集分别保存为CSV文件
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)

# 对于特定的SAEs数据集处理，去除不需要的列，并重命名剩余的列
saes_data = accumulated_data.drop(columns=['Station Length', 'Avg Speed', 'Lane 1 Avg Speed', 'Lane 1 Observed', '% Observed'])
saes_data.rename(columns={
    'Timestamp': '5 Minutes',
    'Lane 1 Flow': 'Lane 1 Flow (Veh/5 Minutes)'
}, inplace=True)
# 类似之前的操作，转换时间列的格式
saes_data['5 Minutes'] = pd.to_datetime(saes_data['5 Minutes'])
saes_data['5 Minutes'] = saes_data['5 Minutes'].dt.strftime('%m/%d/%Y %H:%M')

# 再次分割数据为训练集和测试集，并保存为CSV文件
split_point_saes = int(len(saes_data) * 0.8)
train_data_saes = saes_data[:split_point_saes]
test_data_saes = saes_data[split_point_saes:]

train_data_saes.to_csv('train_saes.csv', index=False)
test_data_saes.to_csv('test_saes.csv', index=False)
```

最后得到的train.csv和test.csv文件用于训练LSTM、GRU模型。此外，train_saes.csv和test_saes.csv用于训练SAEs模型。

#### 1.2.3 建立处理数据的文件

为了处理和准备时间序列数据，以便用于机器学习模型的训练和测试。

将数据读取并进行预处理。为了特征值转换到一个共同的尺度上，需要进行特征缩放，这是必须的步骤，因为算法的性能很容易受到范围差异较大特征数值的影响。为了达到这个要求，简单进行最小-最大归一化。

> 最小-最大归一化：
> 顾名思义，就是利用数据列中的最大值和最小值进行标准化处理，标准化后的数值处于[0,1]之间，计算方式为数据与该列的最小值作差，再除以极差。
> 具体公式为：
> ![image-20240328204238995](https://s2.loli.net/2024/03/28/8H4rgiLX2MJzS6t.png)
>
> 公式中，x’表示单个数据的取值，min是数据所在列的最小值，max是数据所在列的最大值。

将这些处理好的数据进行一定的构造，以适应机器学习，我在此处着重使用了时间滞后（lags）建立数据集，通过使用前11个时间点的数据来预测下一个时间点的值。

根据算法解读，SAEs与另外两种算法的适用数据集不应该相同，SAEs可以收集多个特征进行预算，但另外两种算法应该注意到最重要的flow上，所以需要分为两种方法以适应不同的数据集。

最终生成准备好用于训练和测试机器学习的数据集，包括特征数据（X_train, X_test）和目标数据（y_train, y_test）。

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')  # 尝试读取CSV文件，并使用utf-8编码
        df['5 Minutes'] = pd.to_datetime(df['5 Minutes'])  # 将'5 Minutes'列转换为datetime类型
        df = df.fillna(method='ffill').fillna(method='bfill')  # 先向前填充缺失值，如果仍有缺失则向后填充
        print(f"{file_path} 文件读取完成")  # 打印成功读取文件的消息
        return df  # 返回读取的DataFrame
    except FileNotFoundError as e:  # 如果文件未找到，捕获异常
        print(f"文件未找到: {e}")  # 打印文件未找到的错误消息
        return None  # 返回None
    except Exception as e:  # 捕获其他所有异常
        print(f"读取文件时出现错误: {e}")  # 打印出错信息
        return None  # 返回None

def scale_data(df, attr):
    scaler = MinMaxScaler(feature_range=(0, 1))  # 创建MinMaxScaler对象，设定缩放范围为0到1
    df[attr] = scaler.fit_transform(df[[attr]])  # 对指定属性进行缩放，并替换原来的列
    return df, scaler  # 返回缩放后的DataFrame和缩放器对象

def create_dataset(data, lags, target_data=None):
    X, y = [], []  # 初始化输入和输出列表
    if target_data is None:  # 如果没有提供目标数据
        target_data = data[:, -1]  # 使用输入数据的最后一列作为目标数据
    for i in range(lags, len(data)):  # 循环创建输入和目标数据
        X.append(data[i-lags:i])  # 选择从当前点向前lags个数据点作为输入
        y.append(target_data[i])  # 当前点的目标数据作为输出
    return np.array(X), np.array(y)  # 将列表转换为数组并返回

def process_and_check_data():
    # 处理常规数据集
    X_train, y_train, X_test, y_test, scaler = process_data('data/train.csv', 'data/test.csv', 12)
    if X_train is not None and X_test is not None:  # 检查数据是否成功处理
        print("常规数据处理完成")
        print(f"Training data shape: {X_train.shape}")  # 打印训练数据形状
        print(f"Testing data shape: {X_test.shape}")  # 打印测试数据形状
    else:
        print("常规数据处理失败，跳过打印数据形状。")
    
    # 处理SAEs数据集
    X_train_saes, y_train_saes, X_test_saes, y_test_saes, scaler_saes = process_saes_data('data/train_saes.csv', 'data/test_saes.csv', 12)
    if X_train_saes is not None and X_test_saes is not None:  # 检查SAEs数据是否成功处理
        print("SAEs数据处理完成")
        print(f"Training data SAES shape: {X_train_saes.shape}")  # 打印SAEs训练数据形状
        print(f"Testing data SAES shape: {X_test_saes.shape}")  # 打印SAEs测试数据形状
    else:
        print("SAEs数据处理失败，跳过打印数据形状。")

def process_data(train_file, test_file, lags, attr='Lane 1 Flow (Veh/5 Minutes)'):
    df_train = read_data(train_file)  # 读取训练数据文件
    df_test = read_data(test_file)  # 读取测试数据文件
    if df_train is None or df_test is None:  # 如果文件读取失败
        return None, None, None, None, None  # 返回空值
    df_train, scaler = scale_data(df_train, attr)  # 缩放训练数据
    df_test[attr] = scaler.transform(df_test[[attr]])  # 使用相同的缩放器缩放测试数据
    X_train, y_train = create_dataset(df_train[attr].values.reshape(-1, 1), lags)  # 创建训练数据集
    X_test, y_test = create_dataset(df_test[attr].values.reshape(-1, 1), lags)  # 创建测试数据集
    return X_train, y_train, X_test, y_test, scaler  # 返回数据集和缩放器

def process_saes_data(train_file, test_file, lags):
    df_train = read_data(train_file)  # 读取训练数据文件
    df_test = read_data(test_file)  # 读取测试数据文件
    if df_train is None or df_test is None:  # 如果文件读取失败
        print("SAEs数据处理失败，无法读取文件")
        return None, None, None, None, None, None  # 返回空值
    features = df_train.select_dtypes(include=[np.number]).columns.tolist()  # 选择数值类型的列作为特征
    features.remove('Lane 1 Flow (Veh/5 Minutes)')  # 从特征中移除目标变量列

    # 缩放特征
    features_scaler = MinMaxScaler(feature_range=(0, 1))
    df_train[features] = features_scaler.fit_transform(df_train[features])  # 缩放训练数据的特征
    df_test[features] = features_scaler.transform(df_test[features])  # 缩放测试数据的特征

    # 缩放目标变量
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    df_train['Lane 1 Flow (Veh/5 Minutes)'] = target_scaler.fit_transform(df_train[['Lane 1 Flow (Veh/5 Minutes)']])  # 缩放训练数据的目标变量
    df_test['Lane 1 Flow (Veh/5 Minutes)'] = target_scaler.transform(df_test[['Lane 1 Flow (Veh/5 Minutes)']])  # 缩放测试数据的目标变量

    # 使用缩放过的特征和目标变量创建数据集
    X_train, y_train = create_dataset(df_train[features].values, lags, df_train['Lane 1 Flow (Veh/5 Minutes)'].values)  # 创建训练数据集
    X_test, y_test = create_dataset(df_test[features].values, lags, df_test['Lane 1 Flow (Veh/5 Minutes)'].values)  # 创建测试数据集
    return X_train, y_train, X_test, y_test, features_scaler, target_scaler  # 返回数据集和缩放器

if __name__ == "__main__":
    process_and_check_data()  # 调用函数以开始数据处理流程
```



### 1.3 建立模型

#### 1.3.1 分析模型结构

交通流量预测本质上是一个时间序列问题，需要处理历史数据中的时间依赖性。LSTM和GRU是两种专门为解决时间序列数据问题而设计的循环神经网络（RNN）变体。SAE能够自动学习这些复杂的特征，而不需要人工设计特征，这可以提高模型的泛化能力和预测准确性。

LSTM和GRU可以捕捉时间依赖性，而SAE可以提取深层特征。通过结合这些模型的优势，可能会获得比单一模型更准确的预测结果

##### 1.3.1.1 LSTM（长短期记忆模型）

选择长短期记忆模型的理由非常简单，他在能够学习长期依赖关系，这对于预测未来交通流量来说非常重要，因为交通流量可能受到过去几小时前事件的影响。LSTM提供了高级的时间序列学习能力，但计算成本较高。

在这一次实验中将LSTM分为两个LSTM层和Dropout层，最后是全连接（Dense）层。

###### 1.3.1.1.1 第一层LSTM

该输入层将形状为(units[0], 1)的数据进行接收，units[0]表示时间序列的长度，1表示特征数量。

> 这层的输出维度为units[1]，并设置`return_sequences=True`，意味着会返回每个时间步的输出，为下一层提供序列输入。

###### 1.3.1.1.2 第二层LSTM

接收前一层的序列输出，输出维度为units[2]。这一层不再提供输入，只返回最终输出，用于给全连接层提供数据。

###### 1.3.1.1.3 Dropout层

随机丢弃20%的单元数据，以防止过拟合。

###### 1.3.1.1.4 全连接（Dense）层

输出维度为units[3]，使用sigmoid激活函数，用于最终的回归任务。

> sigmoid激活函数：
>
> - **范围**：Sigmoid函数的输出界限在0和1之间，包含两端。
> - **形状**：它有一个S形曲线（sigmoid曲线）。
> - **输出解释**：接近1的值表示高度激活，而接近0的值表示低激活。这可以直观地理解为神经元被激活的概率。

##### 1.3.1.2 GRU（门控循环单元）

GRU 是LSTM的一个变体，具有更简单的结构，通常计算效率更高。它同样适用于时间序列预测，并且在某些情况下能够与LSTM达到相似的性能。GRU作为一种更轻量级的选项，提供了类似的能力但计算效率更高。

构造分布和LSTM的构造同理。

###### 1.3.1.2.1 第一层GRU

类似于LSTM模型的第一层，接收形状为(units[0], 1)的输入，输出维度为units[1]，并返回每个时间步的输出。

###### 1.3.1.2.2 第二层GRU

接收前一层的序列输出，输出维度为units[2]，只返回序列的最后输出。

###### 1.3.1.2.3 Dropout层

同上，随机丢弃20%的单元。

###### 1.3.1.2.4 全连接（Dense）层

输出维度为units[3]，使用sigmoid激活函数。

##### 1.3.1.3 SAE（堆叠自编码器）

SAE 堆叠自编码器是一种深度神经网络，通过无监督学习逐层预训练来学习输入数据的高阶特征表示。在交通流量预测中，高阶特征可能包括不同时间段的流量模式的变化。SAE通过无监督的方式学习数据特征，可以在不增加额外监督信息的情况下增强模型的特征提取能力。

它的分层很简单：单独的SAE模型、堆叠SAE模型(saes)。

###### 1.3.1.3.1 单独的SAE模型

使用`_get_sae`函数构建，包含一个输入层到隐藏层的全连接层，一个sigmoid激活层，一个Dropout层，和一个从隐藏层到输出层的全连接层。这里构建了三个单独的SAE模型，每个模型针对不同的层级结构。

###### 1.3.1.3.2 堆叠SAE模型(saes)

通过顺序堆叠多个SAE模型的方式构建。首先，为每个隐藏层添加全连接层和sigmoid激活层，然后在最后一个隐藏层后添加Dropout层，最终添加一个输出层的全连接层和sigmoid激活函数。这样，堆叠SAE模型由多个自编码器层按顺序堆叠而成，可以处理更复杂的数据特征。



综上所述，使用不同类型的模型可以探索模型融合的策略，即结合多个模型的预测结果来提高整体预测的准确性。例如，LSTM和GRU可以捕捉时间依赖性，而SAE可以提取深层特征。通过结合这些模型的优势，可能会获得比单一模型更准确的预测结果。



> 为什么三个模型都使用sigmoid激活函数在最后一层：
> 因为最终都是回归问题或二分类问题，sigmoid激活函数符合两种要求，减小代码的健壮度和鲁棒性。

##### 1.3.2 模型构建代码

综上所述，得到以下代码：

```python
# 导入Keras库中的必要模块
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential

def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    构建LSTM模型的函数。

    # 参数
        units: List(int)，输入、输出和隐藏层的单元数。
    # 返回
        model: Model, 神经网络模型。
    """

    model = Sequential()  # 初始化一个顺序模型
    # 向模型中添加一个LSTM层。units[1]表示输出空间的维度，input_shape定义了输入的形状。
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    # 添加另一个LSTM层，units[2]为该层的单元数。这里不设置return_sequences，意味着这层只返回最后一个时间步的输出。
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))  # 添加Dropout层，以0.2的比例随机丢弃一些单元，防止过拟合。
    model.add(Dense(units[3], activation='sigmoid'))  # 添加全连接层，units[3]为该层单元数，激活函数为sigmoid。

    return model  # 返回构建的模型

def get_gru(units):
    """GRU(Gated Recurrent Unit)
    构建GRU模型的函数。

    # 参数
        units: List(int)，输入、输出和隐藏层的单元数。
    # 返回
        model: Model, 神经网络模型。
    """

    model = Sequential()  # 初始化一个顺序模型
    # 向模型中添加一个GRU层。units[1]表示输出空间的维度，input_shape定义了输入的形状。
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    # 添加另一个GRU层，units[2]为该层的单元数。不设置return_sequences，只返回最后一个时间步的输出。
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))  # 添加Dropout层，以0.2的比例随机丢弃一些单元，防止过拟合。
    model.add(Dense(units[3], activation='sigmoid'))  # 添加全连接层，units[3]为该层单元数，激活函数为sigmoid。

    return model  # 返回构建的模型

def _get_sae(inputs, hidden, output):
    """SAE(Self Auto-Encoders)
    构建SAE模型的函数。

    # 参数
        inputs: Integer, 输入单元数。
        hidden: Integer, 隐藏层单元数。
        output: Integer, 输出层单元数。
    # 返回
        model: Model, 神经网络模型。
    """

    model = Sequential()  # 初始化一个顺序模型
    model.add(Dense(hidden, input_dim=inputs, name='hidden'))  # 添加全连接层，hidden为该层单元数，inputs为输入维度。
    model.add(Activation('sigmoid'))  # 添加激活层，使用sigmoid函数。
    model.add(Dropout(0.2))  # 添加Dropout层，以0.2的比例随机丢弃一些单元，防止过拟合。
    model.add(Dense(output, activation='sigmoid'))  # 添加全连接层，output为该层单元数，激活函数为sigmoid。

    return model  # 返回构建的模型

def get_saes(layers):
    """SAEs(Stacked Auto-Encoders)
    构建SAEs模型的函数。

    # 参数
        layers: List(int)，输入、输出和隐藏层的单元数。
    # 返回
        models: List(Model)，SAE和SAEs的列表。
    """
    # 使用给定的层参数构建三个SAE模型
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential()  # 初始化一个顺序模型，用于构建堆叠自编码器
    # 以下是添加不同隐藏层的过程
    saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[2], name='hidden2'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[3], name='hidden3'))
    saes.add(Activation('sigmoid'))
    saes.add(Dropout(0.2))  # 添加Dropout层
    saes.add(Dense(layers[4], activation='sigmoid'))  # 添加输出层

    models = [sae1, sae2, sae3, saes]  # 将单个SAE模型和整个SAEs模型放入列表中

    return models  # 返回模型列表
```



## 2 训练与测试

### 2.1 训练

在训练阶段，模型通过学习训练数据集中的特征和对应的标签（在本案例中是历史交通流量数据及其对应的实际流量），调整其内部参数，以最小化预测误差（即损失函数的值）。这个过程是模型学习如何预测未来交通流量的关键。

#### 2.1.1 LSTM训练

1. **数据预处理**：首先，需要准备和处理时间序列数据，使其适用于深度学习模型的训练。这包括加载数据、归一化、定义输入特征和目标变量等步骤。数据预处理是建立有效模型的关键步骤。

2. **模型构建**：根据选择的模型类型（LSTM、GRU或SAES），构建相应的神经网络模型。每种模型都有其特定的结构和配置。

3. **模型训练和评估**：使用训练数据来训练模型，并通过验证集监控模型的泛化能力。在这个过程中，使用了回调函数（如模型检查点和早停）来优化训练过程，防止过拟合，并保存最佳模型。

4. **结果展示**：最后，绘制训练和验证损失曲线，直观展示模型在训练过程中的表现。通过损失曲线，可以观察模型是否收敛，以及是否存在过拟合或欠拟合的情况

   > **模型检查点（Model Checkpointing）**
   >
   > 模型检查点是一种在训练过程中定期保存模型的技术。这不仅可以在训练过程中意外中断时保护模型的学习进度，还可以帮助我们保存和选择表现最好的模型。使用模型检查点，你可以指定一个监视的指标（例如，验证集上的损失或准确度），并只保存在这个指标上表现最好的模型。
   >
   > 在TensorFlow或Keras中，这通过使用`ModelCheckpoint`回调实现，它允许你定义检查点的存储路径、何时保存模型（每个epoch结束时）、以及基于评估指标来决定是否覆盖上一个检查点。
   >
   > 例如，如果你监视验证集的损失，并设置`save_best_only=True`，那么在验证集损失没有改善时，当前的模型就不会覆盖已保存的模型。这确保了你保存的是整个训练过程中性能最好的模型。
   >
   > **早停（Early Stopping）**
   >
   > 早停是另一种防止神经网络过拟合的技术。通过监视一个指标（例如，验证集上的损失或准确度），如果在一定数量的epochs之后这个指标没有改善，则提前终止训练。这意味着模型不会继续学习到可能导致过拟合的噪声或不重要的模式。
   >
   > 在TensorFlow或Keras中，这通过使用`EarlyStopping`回调实现。你可以指定要监视的指标、没有改善的epochs数量（`patience`参数），以及是否在训练结束时恢复到最佳模型的权重。
   >
   > 使用早停技术可以节省时间和计算资源，并帮助你获得泛化能力更强的模型。

为了一些像我一样刚刚学习深度学习代码的朋友，我将每一行代码进行注释解析。

```python
import sys # 导入sys模块，用于访问与Python解释器紧密相关的变量和函数。
import warnings # 导入warnings模块，用于警告控制。
import argparse # 导入argparse模块，用于命令行参数解析。
import numpy as np # 导入numpy库，并以np为别名，用于数组和矩阵计算。
import pandas as pd # 导入pandas库，并以pd为别名，用于数据处理和分析。
import tensorflow as tf # 导入tensorflow库，并以tf为别名，用于机器学习和深度学习。
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # 从tensorflow.keras.callbacks导入EarlyStopping和ModelCheckpoint类，用于模型训练过程中的回调。
from tensorflow.keras.optimizers import Adam # 从tensorflow.keras.optimizers导入Adam优化器，用于模型训练。
import matplotlib.pyplot as plt # 导入matplotlib.pyplot，并以plt为别名，用于绘制图形。

sys.path.append('/root/method') # 将'/root/method'目录添加到sys.path中，使其成为模块搜索路径的一部分。
from model.model import get_lstm, get_gru, get_saes # 从model.model模块导入get_lstm, get_gru, get_saes函数，这些函数用于获取相应的模型。
from data.data import process_data, process_and_check_data # 从data.data模块导入process_data和process_and_check_data函数，用于数据处理和检查。

warnings.filterwarnings('ignore') # 忽略警告信息，使之不在控制台输出。

# 定义r_squared函数，用于计算R平方值，衡量模型预测值与实际值的相关程度。
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) # 计算残差平方和。
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) # 计算总平方和。
    return (1 - SS_res / (SS_tot + K.epsilon())) # 返回R平方值。

# 定义explained_variance_score函数，用于计算解释方差分数，衡量模型对数据集波动的解释能力。
def explained_variance_score(y_true, y_pred):
    numerator = K.sum(K.square(y_true - K.mean(y_true)) - K.square(y_true - y_pred)) # 计算分子。
    denominator = K.sum(K.square(y_true - K.mean(y_true))) # 计算分母。
    return numerator / (denominator + K.epsilon()) # 返回解释方差分数。

# 定义rmse函数，用于计算均方根误差，衡量模型预测值与实际值的偏差。
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) # 返回均方根误差。

# 定义train_model函数，用于训练模型。
def train_model(model, X_train, y_train, name, config):
    filepath = 'model/' + name + '_best.h5' # 定义模型保存路径。
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') # 创建ModelCheckpoint回调。
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1) # 创建EarlyStopping回调。
    
    callbacks_list = [checkpoint, early_stopping] # 创建回调列表。
    
    # 调用model的fit方法开始训练模型。
    history = model.fit(
        X_train, y_train,
        batch_size=config["batch_size"], # 设置批次大小。
        epochs=config["epochs"], # 设置训练轮次。
        validation_split=0.05, # 设置验证集比例。
        callbacks=callbacks_list # 设置回调函数列表。
    )

    model.save('model/' + name + '.h5') # 保存模型。
    return history # 返回训练历史对象。

# 定义模型训练的配置字典。
config = {
    "batch_size": 128, # 批次大小。
    "epochs": 900, # 训练轮次。
    "lag": 12, # 时间滞后步数。
    "train_file": 'data/train.csv', # 训练数据文件路径。
    "test_file": 'data/test.csv', # 测试数据文件路径。
    "model_configs": { # 模型配置。
        "lstm": [None, 64, 64, 1], # LSTM模型配置。
        "gru": [None, 64, 64, 1], # GRU模型配置。
        "saes": [None, 400, 400, 400, 1] # SAES模型配置。
    }
}

# 调用process_data函数处理训练和测试数据。
X_train, y_train, X_test, y_test, scaler = process_data(config["train_file"], config["test_file"], config["lag"])

process_and_check_data() # 调用process_and_check_data函数检查数据。

model_name = 'lstm'  # 选择模型类型。

# 根据模型类型对训练数据进行维度变换，以适应模型输入要求。
if model_name in ['lstm', 'gru']:
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model_config = config["model_configs"][model_name] # 获取模型配置。
model_config[0] = X_train.shape[1]  # 设置模型输入的时间步长。

# 根据模型名称选择并创建模型。
if model_name == 'lstm':
    model = get_lstm(model_config)
elif model_name == 'gru':
    model = get_gru(model_config)
elif model_name == 'saes':
    models = get_saes(model_config)
    history = train_saes(models, X_train, y_train, model_name, config)
else:
    raise ValueError(f"Unknown model type: {model_name}") # 如果模型类型未知，抛出异常。

# 如果是LSTM或GRU模型，调用train_model函数进行训练。
if model_name in ['lstm', 'gru']:
    history = train_model(model, X_train, y_train, model_name, config)

# 绘制训练和验证过程中的损失曲线。
plt.figure(figsize=(10, 6)) # 设置图形大小。
plt.plot(history.history['loss'], label='Training Loss') # 绘制训练损失曲线。
plt.plot(history.history['val_loss'], label='Validation Loss') # 绘制验证损失曲线。
plt.title('Training and Validation Loss') # 设置图形标题。
plt.xlabel('Epochs') # 设置x轴标签。
plt.ylabel('Loss') # 设置y轴标签。
plt.legend() # 显示图例。
plt.show() # 显示图形。
```

#### 2.1.2 GRU训练

1. **准备环境**：通过导入必要的库和模块，准备代码运行的环境。这包括TensorFlow、NumPy、matplotlib等，用于构建模型、处理数据和绘图。
2. **定义评估函数**：自定义模型评估指标，如RMSE（均方根误差）和R²（决定系数），以便在训练过程中监控模型的性能。
3. **构建和训练GRU模型**：定义一个函数来配置、编译和训练GRU模型，其中包括模型的保存逻辑，以及使用回调函数（如早停和模型检查点）来优化训练过程。
4. **绘制损失曲线**：通过另一个函数，绘制训练过程中训练集和验证集的损失曲线，以便于观察模型的学习进度和过拟合情况。
5. **主函数流程**：定义一个主函数，用于执行上述所有步骤。这包括配置参数、数据预处理、模型训练和结果可视化。

```python
import sys # 导入sys库，用于访问由Python解释器管理的变量和函数。
import numpy as np # 导入NumPy库，并命名为np，用于高效的多维数组操作。
import tensorflow as tf # 导入TensorFlow库，并命名为tf，用于机器学习和神经网络模型的构建和训练。
import matplotlib.pyplot as plt # 导入matplotlib库的pyplot模块，并命名为plt，用于数据可视化。
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # 从tensorflow.keras.callbacks导入EarlyStopping和ModelCheckpoint类，用于训练时的回调控制。
from tensorflow.keras.optimizers import Adam # 从tensorflow.keras.optimizers导入Adam优化器，用于模型训练过程中的参数优化。
from model.model import get_gru # 从model.model模块导入get_gru函数，用于获取GRU模型的实例。
from data.data import process_data # 从data.data模块导入process_data函数，用于处理输入数据。

# 定义rmse评估函数，用于计算预测值和真实值之间的均方根误差。
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# 定义r_squared评估函数，用于计算预测值的确定系数，评估模型拟合的好坏。
def r_squared(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred)) # 计算残差平方和
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) # 计算总平方和
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon())) # 返回确定系数

# 定义训练GRU模型的函数。
def train_gru(X_train, y_train, config, model_name='gru'):
    model_config = config["model_configs"][model_name] # 获取模型配置。
    model_config[0] = X_train.shape[1] # 设置模型的时间步长。
    model = get_gru(model_config) # 创建GRU模型实例。

    filepath = f'model/{model_name}_best.h5' # 设置模型保存路径。
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') # 设置模型检查点回调函数。
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1) # 设置早停回调函数，以防过拟合。

    # 编译模型，设置优化器、损失函数和评估指标。
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mse', rmse, 'mae', r_squared])

    # 训练模型，并设置批次大小、训练轮次、验证集比例和回调函数列表。
    history = model.fit(
        X_train, y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_split=0.05,
        callbacks=[checkpoint, early_stopping]
    )

    model.save(f'model/{model_name}.h5') # 保存训练好的模型。
    return history # 返回训练历史记录。

# 定义绘制训练和验证损失曲线的函数。
def plot_loss(history):
    plt.figure(figsize=(10, 6)) # 设置图形大小。
    plt.plot(history.history['loss'], label='Training Loss') # 绘制训练损失曲线。
    plt.plot(history.history['val_loss'], label='Validation Loss') # 绘制验证损失曲线。
    plt.title('Training and Validation Loss') # 设置图形标题。
    plt.xlabel('Epochs') # 设置x轴标签为"Epochs"。
    plt.ylabel('Loss') # 设置y轴标签为"Loss
	plt.legend() # 显示图例。
	plt.show() # 显示图形。
```

#### 2.1.3 SAEs训练

1.数据预处理

在机器学习项目中，数据预处理是一个关键步骤。通过`process_saes_data`函数处理数据，你的目标可能是格式化数据，使其适合SAEs模型的输入要求。这包括特征缩放、将数据转换为适合模型输入的格式等。成功处理数据是确保模型能够学习到有效特征的前提。

2.模型训练与评估

模型训练是迭代过程，通过不断调整内部参数来最小化预测误差。使用`EarlyStopping`和`ModelCheckpoint`回调函数可以在合适的时候自动停止训练并保存最佳模型，这有助于防止过拟合并节省时间。自定义的评估函数（如RMSE和R-squared）可以提供关于模型性能的具体信息。

3.结果分析和调优

通过可视化训练和验证损失，可以评估模型的学习过程，识别过拟合或欠拟合的情况，并据此调整模型参数或训练策略。

```python
# 导入必要的库
import sys  # 导入sys库，可用于处理一些系统相关的操作
import numpy as np  # 导入numpy库，用于进行高效的矩阵和数组操作
import tensorflow as tf  # 导入tensorflow库，一个强大的机器学习库
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot，用于绘制图形

# 从tensorflow.keras.callbacks导入EarlyStopping和ModelCheckpoint，用于模型训练时的回调
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam  # 从tensorflow.keras.optimizers导入Adam，一个优化器，用于模型训练
from model.model import get_saes  # 从model.model导入get_saes函数，假定这是一个用于获取模型结构的函数
from data.data import process_saes_data  # 从data.data导入process_saes_data函数，假定这是一个用于处理数据的函数

# 自定义的评估函数
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))  # 计算均方根误差(RMSE)，评估模型性能

def r_squared(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))  # 计算残差平方和
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))  # 计算总平方和
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))  # 计算R-squared值，评估模型拟合度

def flatten_input_data(X):
    """将输入数据从三维 (samples, timesteps, features) 展平为二维 (samples, timesteps*features)"""
    return X.reshape((X.shape[0], np.prod(X.shape[1:])))  # 使用reshape函数和np.prod将输入数据展平

# 训练SAEs模型的函数
def train_saes(X_train, y_train, config, model_name='saes'):
    model_config = config["model_configs"][model_name]  # 从配置中获取模型配置
    model_config[0] = X_train.shape[1]  # 设置模型的第一个维度为输入数据的时间步长
    models = get_saes(model_config)  # 根据模型配置获取SAEs模型结构

    for i, model in enumerate(models):  # 遍历模型列表，对每个模型进行训练
        filepath = f'model/{model_name}_model_{i}_best.h5'  # 设置模型保存路径
        # 创建ModelCheckpoint回调，用于保存验证损失最小的模型
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # 创建EarlyStopping回调，用于在验证损失不再改善时提前停止训练
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)

        # 编译模型，设置优化器、损失函数和评估指标
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mse', rmse, 'mae', r_squared])

        # 训练模型，并设置批次大小、迭代次数、验证集比例和回调函数
        history = model.fit(
            X_train, y_train,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            validation_split=0.05,
            callbacks=[checkpoint, early_stopping]
        )

        # 保存训练完成的模型
        model.save(f'model/{model_name}_model_{i}.h5')

    return models  # 返回训练好的模型列表

def main():
    # 配置参数
    config = {
        "batch_size": 128,  # 批次大小
        "epochs": 900,  # 迭代次数
        "lag": 12,  # 时间滞后
        "train_file": 'data/train_saes.csv',  # 训练文件路径
        "test_file": 'data/test_saes.csv',  # 测试文件路径
        "model_configs": {
        "saes": [96, 400, 400, 400, 1]  # 模型配置，假设处理后的数据维度是96
        }
    }

    try:
        # 使用process_saes_data函数处理数据，并接收返回的值
        X_train_saes, y_train_saes, X_test_saes, y_test_saes, features_scaler_saes, target_scaler_saes = process_saes_data(
            config["train_file"], config["test_file"], config["lag"]
        )
        if X_train_saes is None or X_test_saes is None or y_train_saes is None or y_test_saes is None:
            print("数据处理失败，一些数据为空，请检查文件路径和名称是否正确")
            return
        else:
            print(f"数据处理成功，训练数据形状: {X_train_saes.shape}, 训练标签形状: {y_train_saes.shape}")
    except Exception as e:
        print(f"处理SAEs数据时出错: {e}")  # 捕获并打印异常信息
        return

    input_dim = X_train_saes.shape[1] * X_train_saes.shape[2]  # 计算输入维度
    models = get_saes(input_dim)  # 根据输入维度获取模型结构

    # 选择最后一个模型进行训练
    saes_model = models[-1]
    print("开始训练SAEs模型...")
    
    # 确保输入数据是正确的形状
    X_train_saes_flat = X_train_saes.reshape((X_train_saes.shape[0], -1))
    history = saes_model.fit(
        X_train_saes_flat, y_train_saes,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=50, verbose=1),
            ModelCheckpoint('model/saes_best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
        ]
    )

    print("SAEs模型训练完成")

    # 绘制训练和验证的损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')  # 设置图表标题
    plt.xlabel('Epoch')  # 设置x轴标签为"Epoch"
    plt.ylabel('Loss')  # 设置y轴标签为"Loss"
    plt.legend()  # 显示图例
    plt.show()  # 显示图表

if __name__ == '__main__':
    main()  # 如果直接运行此脚本，则执行main函数
```



### 2.2 测试

在测试阶段，使用与训练阶段不同的数据集（测试数据集）来评估模型的性能和泛化能力。这可以帮助我们了解模型在未见过的数据上的表现如何，验证模型是否过拟合或欠拟合，以及模型的实际预测能力。

#### 2.2.1 LSTM测试

用于加载已经训练好的LSTM模型，对测试数据进行预测，并使用多种指标评估模型性能。包括自定义的评估指标和使用`sklearn`及`TensorFlow`的内建函数来计算常用的统计量，最后将结果打印输出。

```python
import tensorflow as tf  # 导入TensorFlow库，这是一个用于深度学习的开源库。
from tensorflow.keras.models import load_model  # 从tensorflow.keras.models导入load_model函数，用于加载之前训练好的模型。
from data.data import process_data  # 从data包的data模块导入process_data函数，假设这个函数用于数据处理。
import numpy as np  # 导入numpy库，一个用于高效数学运算的库，常用于数组处理。
from tensorflow.keras import backend as K  # 从tensorflow.keras导入backend模块并重命名为K，这是TensorFlow的后端接口，提供了一系列底层操作的函数。
from sklearn.metrics import mean_absolute_error, explained_variance_score  # 从sklearn.metrics导入mean_absolute_error和explained_variance_score函数，用于评估模型性能。

# 自定义的评估函数
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))  # 定义均方根误差(RMSE)函数，用于评估模型预测值和实际值之间的误差。

def r_squared(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # 将y_true转换为float32类型，确保数值计算的类型一致。
    y_pred = tf.cast(y_pred, tf.float32)  # 将y_pred转换为float32类型。
    SS_res = K.sum(K.square(y_true - y_pred))  # 计算残差平方和SS_res。
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))  # 计算总平方和SS_tot。
    return (1 - SS_res / (SS_tot + K.epsilon()))  # 计算R-squared值，K.epsilon()用于防止分母为0。

# 修改后的MAPE函数
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)  # 将输入转换为numpy数组。
    non_zero = np.where(y_true != 0)  # 找出实际值非零的索引。
    y_true_non_zero = y_true[non_zero]  # 获取非零的实际值。
    y_pred_non_zero = y_pred[non_zero]  # 获取对应的预测值。
    return np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100  # 计算MAPE值。

# 加载模型，并确保使用自定义评估函数
model = load_model('model/lstm_best.h5', custom_objects={'rmse': rmse, 'r_squared': r_squared})  # 加载模型，指定自定义的评估函数。

# 假设process_data函数已经被定义并可以正常使用
_, _, X_test, y_test, scaler = process_data('data/train.csv', 'data/test.csv', 12)  # 使用process_data函数处理数据，得到测试集和标签，以及用于数据缩放的scaler对象。

# 确保X_test是三维的，适合LSTM或GRU模型
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # 重塑X_test为三维数组，以适应LSTM或GRU模型的输入要求。

# 进行预测
y_pred = model.predict(X_test)  # 使用模型对测试集进行预测。

# 将预测和实际标签的缩放逆转（使用了缩放）
y_pred_rescaled = scaler.inverse_transform(y_pred)  # 使用scaler的inverse_transform方法将预测值的缩放逆转。
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))  # 将测试标签的缩放逆转。

# 评估性能
mse = tf.keras.metrics.mean_squared_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten()).numpy()  # 计算均方误差(MSE)。
rmse_value = rmse(tf.constant(y_test_rescaled.flatten(), dtype=tf.float32), tf.constant(y_pred_rescaled.flatten(), dtype=tf.float32)).numpy()  # 计算RMSE值。
r2_value = r_squared(tf.constant(y_test_rescaled.flatten(), dtype=tf.float32), tf.constant(y_pred_rescaled.flatten(), dtype=tf.float32)).numpy()  # 计算R-squared值。
mae_value = mean_absolute_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten())  # 计算平均绝对误差(MAE)。
explained_variance = explained_variance_score(y_test_rescaled.flatten(), y_pred_rescaled.flatten())  # 计算解释方差分数。
mape_value = mape(y_test_rescaled.flatten(), y_pred_rescaled.flatten())  # 计算MAPE值。

# 打印评估结果
print(f"测试集 MSE: {mse}")  # 打印均方误差。
print(f"测试集 RMSE: {rmse_value}")  # 打印均方根误差。
print(f"测试集 R-squared: {r2_value}")  # 打印R-squared值。
print(f"测试集 MAE: {mae_value}")  # 打印平均绝对误差。
print(f"测试集 Explained Variance Score: {explained_variance}")  # 打印解释方差分数。
print(f"测试集 MAPE: {mape_value} %")  # 打印平均绝对百分比误差。
```

#### 2.2.2 GRU测试

通过各种评估指标全面地评估了一个模型在测试集上的表现，指标包括MSE、RMSE、R平方值、MAE、解释方差分数和MAPE，这样可以从多个角度理解模型的性能。

```python
# 导入numpy库，通常用于数组和数学运算
import numpy as np
# 导入tensorflow库，一个广泛使用的机器学习和深度学习框架
import tensorflow as tf
# 从tensorflow.keras.models模块导入load_model函数，用于加载保存的模型
from tensorflow.keras.models import load_model
# 从data.data模块导入process_data函数，假设这是用于数据预处理的函数
from data.data import process_data
# 导入sklearn.metrics模块中的mean_absolute_error和explained_variance_score函数，用于评估模型性能
from sklearn.metrics import mean_absolute_error, explained_variance_score

# 自定义的评估函数rmse，计算预测值与实际值之间的均方根误差
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# 自定义的评估函数r_squared，计算预测值与实际值之间的R平方值，用于衡量模型的解释能力
def r_squared(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

# 自定义的评估函数mape，计算预测值与实际值之间的平均绝对百分比误差
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = np.where(y_true != 0)  # 找出实际值非零的索引
    y_true_non_zero = y_true[non_zero]
    y_pred_non_zero = y_pred[non_zero]
    return np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100

# 加载模型，模型文件位于'model/gru_best.h5'，同时指定自定义评估函数
model = load_model('model/gru_best.h5', custom_objects={'rmse': rmse, 'r_squared': r_squared})

# 使用process_data函数处理数据，获取测试数据等，假设这个函数负责数据的预处理和分割
_, _, X_test, y_test, scaler = process_data('data/train.csv', 'data/test.csv', 12)

# 确保测试数据X_test的维度适合GRU模型，即三维
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 使用模型对测试数据进行预测
y_pred = model.predict(X_test)

# 将预测值和实际标签值通过scaler逆缩放，还原到原始尺度
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# 使用mean_squared_error函数计算测试集的均方误差(MSE)
mse = tf.keras.metrics.mean_squared_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten()).numpy()
# 使用自定义的rmse函数计算测试集的均方根误差(RMSE)
rmse_value = rmse(tf.constant(y_test_rescaled.flatten(), dtype=tf.float32), tf.constant(y_pred_rescaled.flatten(), dtype=tf.float32)).numpy()
# 使用自定义的r_squared函数计算测试集的R平方值
r2_value = r_squared(tf.constant(y_test_rescaled.flatten(), dtype=tf.float32), tf.constant(y_pred_rescaled.flatten(), dtype=tf.float32)).numpy()
# 使用mean_absolute_error函数计算测试集的平均绝对误差(MAE)
mae_value = mean_absolute_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten())
# 使用explained_variance_score函数计算测试集的解释方差分数
explained_variance = explained_variance_score(y_test_rescaled.flatten(), y_pred_rescaled.flatten())
# 使用自定义的mape函数计算测试集的平均绝对百分比误差(MAPE)
mape_value = mape(y_test_rescaled.flatten(), y_pred_rescaled.flatten())

# 打印出模型在测试集上的各项性能指标
print(f"测试集 MSE: {mse}")  # 打印均方误差
print(f"测试集 RMSE: {rmse_value}")  # 打印均方根误差
print(f"测试集 R-squared: {r2_value}")  # 打印R平方值
print(f"测试集 MAE: {mae_value}")  # 打印平均绝对误差
print(f"测试集 Explained Variance Score: {explained_variance}")  # 打印解释方差分数
print(f"测试集 MAPE: {mape_value} %")  # 打印平均绝对百分比误差
```

#### 2.2.3 SAEs测试

```python
import numpy as np  # 导入numpy库，用于数学计算和矩阵操作
import tensorflow as tf  # 导入tensorflow库，用于机器学习和神经网络建模
from tensorflow.keras.models import load_model  # 从tensorflow.keras.models导入load_model函数，用于加载训练好的模型
from data.data import process_saes_data  # 从data包的data模块导入process_saes_data函数，用于处理SAES数据集
from sklearn.metrics import mean_absolute_error, explained_variance_score  # 从sklearn.metrics导入评估函数，包括平均绝对误差和解释方差得分
from sklearn.preprocessing import MinMaxScaler  # 从sklearn.preprocessing导入MinMaxScaler类，用于数据的最小-最大规范化

# 自定义的评估函数
def rmse(y_true, y_pred):  # 定义均方根误差（RMSE）计算函数
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))  # 返回y_pred与y_true之差的平方的平均值的平方根

def r_squared(y_true, y_pred):  # 定义R-squared（决定系数）计算函数
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))  # 计算残差平方和
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))  # 计算总平方和
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))  # 返回1减去残差平方和除以总平方和加上一个小的epsilon防止除以零

def mape(y_true, y_pred):  # 定义平均绝对百分比误差（MAPE）计算函数
    y_true, y_pred = np.array(y_true), np.array(y_pred)  # 将输入转换为numpy数组
    non_zero = np.where(y_true != 0)  # 找出y_true中非零元素的索引
    y_true_non_zero = y_true[non_zero]  # 获取y_true中非零元素
    y_pred_non_zero = y_pred[non_zero]  # 获取y_pred中对应于y_true非零元素的预测值
    return np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100  # 计算MAPE

# process_saes_data
# 注意：需要确保process_saes_data函数返回处理后的数据集，并正确处理文件名
X_train, y_train, X_test, y_test, features_scaler_saes, target_scaler_saes = process_saes_data('data/train_saes.csv', 'data/test_saes.csv', 12)  # 调用process_saes_data函数处理SAES数据集

# 创建一个新的scaler对象用于Y值
scaler_y = MinMaxScaler()  # 实例化MinMaxScaler
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))  # 对y_train进行规范化并训练Scaler
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))  # 使用相同的Scaler对y_test进行规范化

# 加载模型
model = load_model('model/saes_best_model.h5', custom_objects={'rmse': rmse, 'r_squared': r_squared})  # 加载训练好的模型，同时指定自定义评估函数

# 预处理X_test以符合模型的输入要求
X_test_flat = X_test.reshape((X_test.shape[0], -1))  # 将X_test的形状改变，以适应模型的输入格式

# 进行预测
y_pred = model.predict(X_test_flat)  # 使用模型对处理后的X_test进行预测

# 使用针对y值训练的scaler来逆变换预测结果和测试集标签
y_pred_rescaled = scaler_y.inverse_transform(y_pred)  # 使用y的Scaler逆变换模型的预测结果
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))  # 使用y的Scaler逆变换测试集的y值

# 评估性能
mse = tf.keras.metrics.mean_squared_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten()).numpy()  # 计算测试集的均方误差(MSE)
rmse_value = rmse(tf.constant(y_test_rescaled.flatten(), dtype=tf.float32), tf.constant(y_pred_rescaled.flatten(), dtype=tf.float32)).numpy()  # 计算测试集的RMSE
r2_value = r_squared(tf.constant(y_test_rescaled.flatten(), dtype=tf.float32), tf.constant(y_pred_rescaled.flatten(), dtype=tf.float32)).numpy()  # 计算测试集的R-squared值
mae_value = mean_absolute_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten())  # 计算测试集的平均绝对误差(MAE)
explained_variance = explained_variance_score(y_test_rescaled.flatten(), y_pred_rescaled.flatten())  # 计算测试集的解释方差得分
mape_value = mape(y_test_rescaled.flatten(), y_pred_rescaled.flatten())  # 计算测试集的MAPE

print(f"测试集 MSE: {mse}")  # 打印测试集的均方误差
print(f"测试集 RMSE: {rmse_value}")  # 打印测试集的均方根误差
print(f"测试集 R-squared: {r2_value}")  # 打印测试集的R-squared值
print(f"测试集 MAE: {mae_value}")  # 打印测试集的平均绝对误差
print(f"测试集 Explained Variance Score: {explained_variance}")  # 打印测试集的解释方差得分
print(f"测试集 MAPE: {mape_value} %")  # 打印测试集的平均绝对百分比误差
```



## 3 生成最终结果和对比图

设置一个main函数生成最终结果集，并且与数据集进行对比。

### 3.1 mian函数生成最终数据

总体思路和测试时无异。但是只提取20%的数据进行比较。

```python
# 导入所需的库和模块
import numpy as np  # 导入numpy库，用于进行科学计算
import pandas as pd  # 导入pandas库，用于数据处理和分析
import tensorflow as tf  # 导入tensorflow库，用于建立和训练模型
from tensorflow.keras.models import load_model  # 从tensorflow.keras中导入load_model函数，用于加载训练好的模型
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot，用于数据可视化
import sklearn.metrics as metrics  # 导入sklearn的metrics模块，用于计算模型评估指标
from data.data import process_data, process_saes_data  # 从data模块导入process_data和process_saes_data函数，用于数据预处理
from sklearn.preprocessing import MinMaxScaler  # 导入MinMaxScaler，用于进行数据归一化处理

# 定义计算均方根误差的函数
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))  # 计算均方根误差

# 定义计算平均绝对百分比误差的函数
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)  # 将输入转换为numpy数组
    non_zero = y_true != 0  # 找出y_true中非零的元素
    y_true_non_zero = y_true[non_zero]  # 获取非零元素的y_true值
    y_pred_non_zero = y_pred[non_zero]  # 获取对应的y_pred值
    return np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100  # 计算MAPE

# 定义计算决定系数(R^2)的函数
def r_squared(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))  # 计算残差平方和
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))  # 计算总平方和
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))  # 计算并返回R^2

# 定义对回归模型进行评估的函数
def eva_regress(y_true, y_pred):
    vs = metrics.explained_variance_score(y_true, y_pred)  # 计算解释方差分数
    mae = metrics.mean_absolute_error(y_true, y_pred)  # 计算平均绝对误差
    mse = metrics.mean_squared_error(y_true, y_pred)  # 计算均方误差
    r2 = metrics.r2_score(y_true, y_pred)  # 计算R^2
    print(f'explained_variance_score: {vs}')  # 打印解释方差分数
    print(f'mape: {MAPE(y_true, y_pred)}%')  # 打印MAPE
    print(f'mae: {mae}')  # 打印平均绝对误差
    print(f'mse: {mse}')  # 打印均方误差
    print(f'rmse: {np.sqrt(mse)}')  # 打印均方根误差
    print(f'r2: {r2}')  # 打印R^2

# 定义绘图函数，传入时间序列作为x轴
def plot_results(time_series, y_true, y_preds, names, selected_names=None):
    plt.figure(figsize=(15, 5))  # 设置图像大小
    num_points_to_plot = len(y_true) // 5  # 取20%的数据点用于绘图
    time_series = time_series[:num_points_to_plot]  # 获取用于绘图的时间序列数据
    y_true = y_true[:num_points_to_plot]  # 获取用于绘图的真实数据
    plt.plot(time_series, y_true, label='True Data', color='black', linestyle='--')  # 绘制真实数据曲线
    
    if selected_names is None:  # 如果未指定selected_names，则使用所有名称
        selected_names = names
        
    for name, y_pred in zip(names, y_preds):  # 遍历模型名称和预测数据
        if name in selected_names:  # 如果该模型被选中
            y_pred = y_pred[:num_points_to_plot]  # 获取用于绘图的预测数据
            plt.plot(time_series, y_pred, label=name)  # 绘制预测数据曲线
    
    plt.title('Traffic Flow Prediction')  # 设置图标题
    plt.xlabel('Time')  # 设置x轴标签
    plt.ylabel('Flow')  # 设置y轴标签
    plt.gcf().autofmt_xdate()  # 自动格式化日期显示
    plt.legend()  # 显示图例
    plt.show()  # 显示图像

def plot_selected_results(y_true, y_preds, selected_names):
    plt.figure(figsize=(15, 5))  # 设置图像大小
    plt.plot(y_true, label='True Data', color='black', linestyle='--')  # 绘制真实数据曲线
    colors = ['blue', 'green']  # 设置预测数据曲线的颜色
    for name, y_pred, color in zip(selected_names, y_preds, colors):  # 遍历选定的模型名称、预测数据和颜色
        if name in selected_names:  # 如果该模型被选中
            plt.plot(y_pred, label=name, color=color)  # 绘制预测数据曲线
    plt.title('Traffic Flow Prediction (Selected Models)')  # 设置图标题
    plt.xlabel('Time')  # 设置x轴标签
    plt.ylabel('Flow')  # 设置y轴标签
    plt.legend()  # 显示图例
    plt.show()  # 显示图像
    
# 定义主函数
def main():
    config = {
        "batch_size": 128,  # 设置批量大小
        "epochs": 900,  # 设置训练周期数
        "lag": 12,  # 设置时间序列的滞后期数
        "train_file": 'data/train_saes.csv',  # 训练数据文件路径
        "test_file": 'data/test_saes.csv',  # 测试数据文件路径
        "freq": '5T',  # 设置时间序列数据的频率
        "model_configs": {
            "saes": [96, 400, 400, 400, 1]  # SAES模型的配置参数
        }
    }

    y_preds = []  # 用于存储所有模型的预测结果
    models = ['lstm', 'gru', 'saes']  # 定义使用的模型列表
    model_filenames = {  # 定义模型文件名的字典
        'lstm': 'lstm_best.h5',  # LSTM模型文件名
        'gru': 'gru_best.h5',  # GRU模型文件名
        'saes': 'saes_best_model.h5'  # SAES模型文件名
    }
    
    start_time_str = "01/26/2024 14:20"  # 定义测试数据开始时间
    interval_minutes = 5  # 定义时间序列数据的时间间隔
    
    for model_name in models:  # 遍历每个模型
        model_path = f'model/{model_filenames[model_name]}'  # 获取模型文件路径
        model = load_model(model_path, custom_objects={'rmse': rmse, 'r_squared': r_squared})  # 加载模型

        if model_name == 'saes':  # 如果当前模型是SAES
            # 调用process_saes_data函数处理SAES模型的数据
            X_train_saes, y_train_saes, X_test_saes, y_test_saes, features_scaler_saes, target_scaler_saes = process_saes_data(
                config["train_file"], config["test_file"], config["lag"]
            )
            X_test_saes_flat = X_test_saes.reshape((X_test_saes.shape[0], -1))  # 将测试数据扁平化
            y_pred_saes = model.predict(X_test_saes_flat)  # 使用模型进行预测
            y_pred_saes_rescaled = target_scaler_saes.inverse_transform(y_pred_saes)  # 将预测结果反归一化
            pd.DataFrame(y_pred_saes_rescaled.flatten()).to_csv(f'data/pred_{model_name}.csv', index=False, header=[f"{model_name.upper()}_Prediction"])  # 将预测结果保存到CSV文件
            print(f"SAEs模型预测结果已保存到CSV文件。")  # 打印保存成功的消息
            y_preds.append(y_pred_saes_rescaled.flatten())  # 将预测结果添加到列表中
            print(f'Model: {model_name}')  # 打印模型名称
            eva_regress(y_test_saes.flatten(), y_pred_saes_rescaled.flatten())  # 调用eva_regress函数评估模型
        else:
            # 对于非SAEs模型的数据处理
            X_train, y_train, X_test, y_test, scaler = process_data(config["train_file"], config["test_file"], config["lag"])
            X_test_transformed = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # 转换测试数据的形状
            y_pred = model.predict(X_test_transformed)  # 使用模型进行预测
            y_pred_rescaled = scaler.inverse_transform(y_pred)  # 将预测结果反归一化
            y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))  # 将测试数据的真实值反归一化
            pd.DataFrame(y_pred_rescaled.flatten()).to_csv(f'data/pred_{model_name}.csv', index=False, header=[f"{model_name.upper()}_Prediction"])  # 将预测结果保存到CSV文件
            y_preds.append(y_pred_rescaled.flatten())  # 将预测结果添加到列表中
            print(f'Model: {model_name}')  # 打印模型名称
            eva_regress(y_test_rescaled.flatten(), y_pred_rescaled.flatten())  # 调用eva_regress函数评估模型

        # 假设测试集的开始时间已知，并设置为01/26/2024 14:20
    
    # 生成x轴的时间序列
    start_time = pd.to_datetime(start_time_str)  # 将开始时间字符串转换为时间对象
    time_series = [start_time + pd.Timedelta(minutes=i * interval_minutes) for i in range(len(y_test_rescaled.flatten()))]  # 根据时间间隔生成时间序列列表
    
    # 定义函数保存带时间戳的预测结果
    def save_predictions_with_timestamps(predictions, timestamps, filename):
        predictions_with_timestamps = pd.DataFrame({
            'Timestamp': timestamps,  # 时间戳
            'Prediction': predictions  # 预测值
        })
        predictions_with_timestamps.to_csv(filename, index=False)  # 将数据保存为CSV文件
        
    timestamps = [start_time + pd.Timedelta(minutes=i*interval_minutes) for i in range(len(y_test_rescaled))]  # 生成时间戳列表
    
    for i, model_name in enumerate(models):  # 遍历模型列表
        save_predictions_with_timestamps(y_preds[i], timestamps, f'data/pred_{model_name}.csv')  # 保存每个模型的预测结果和时间戳到CSV文件

    # 为绘图选择20%的测试集数据
    num_points_to_plot = len(y_test_rescaled.flatten()) // 5  # 计算绘图所需的数据点数
    y_true = y_test_rescaled.flatten()[:num_points_to_plot]  # 获取用于绘图的真实数据
    time_series = time_series[:num_points_to_plot]  # 获取用于绘图的时间序列数据

    # 绘制所有模型的结果
    plot_results(time_series, y_true, [pred[:num_points_to_plot] for pred in y_preds], models)  # 调用绘图函数

    # 绘制选定模型的结果
    selected_models = ['lstm', 'gru']  # 定义选定的模型列表
    selected_preds = [y_preds[models.index(m)][:num_points_to_plot] for m in selected_models]  # 获取选定模型的预测结果
    plot_results(time_series, y_true, selected_preds, selected_models)  # 调用绘图函数

# 如果此脚本作为主程序运行，则执行main函数
if __name__ == '__main__':
    main()  # 运行主函数
```

### 3.2 对比图

![image-20240328215734723](https://s2.loli.net/2024/03/28/yZJlNFD4rhafesK.png)
