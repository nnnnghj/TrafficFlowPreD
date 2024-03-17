import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df['5 Minutes'] = pd.to_datetime(df['5 Minutes'])
        df = df.fillna(method='ffill').fillna(method='bfill')
        print(f"{file_path} 文件读取完成")
        return df
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return None
    except Exception as e:
        print(f"读取文件时出现错误: {e}")
        return None

def scale_data(df, attr):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[attr] = scaler.fit_transform(df[[attr]])
    return df, scaler

def create_dataset(data, lags, target_data=None):
    X, y = [], []
    if target_data is None:
        target_data = data[:, -1]  # 默认使用最后一列作为目标变量
    for i in range(lags, len(data)):
        X.append(data[i-lags:i])
        y.append(target_data[i])
    return np.array(X), np.array(y)

def process_and_check_data():
    X_train, y_train, X_test, y_test, scaler = process_data('data/train.csv', 'data/test.csv', 12)
    if X_train is not None and X_test is not None:
        print("常规数据处理完成")
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
    else:
        print("常规数据处理失败，跳过打印数据形状。")

    X_train_saes, y_train_saes, X_test_saes, y_test_saes, scaler_saes = process_saes_data('data/train_saes.csv', 'data/test_saes.csv', 12)
    if X_train_saes is not None and X_test_saes is not None:
        print("SAEs数据处理完成")
        print(f"Training data SAES shape: {X_train_saes.shape}")
        print(f"Testing data SAES shape: {X_test_saes.shape}")
    else:
        print("SAEs数据处理失败，跳过打印数据形状。")


def process_data(train_file, test_file, lags, attr='Lane 1 Flow (Veh/5 Minutes)'):
    df_train = read_data(train_file)
    df_test = read_data(test_file)
    if df_train is None or df_test is None:
        return None, None, None, None, None
    df_train, scaler = scale_data(df_train, attr)
    df_test[attr] = scaler.transform(df_test[[attr]])
    X_train, y_train = create_dataset(df_train[attr].values.reshape(-1, 1), lags)
    X_test, y_test = create_dataset(df_test[attr].values.reshape(-1, 1), lags)
    return X_train, y_train, X_test, y_test, scaler

def process_saes_data(train_file, test_file, lags):
    df_train = read_data(train_file)
    df_test = read_data(test_file)
    if df_train is None or df_test is None:
        print("SAEs数据处理失败，无法读取文件")
        return None, None, None, None, None, None
    features = df_train.select_dtypes(include=[np.number]).columns.tolist()
    features.remove('Lane 1 Flow (Veh/5 Minutes)')

    # 为特征创建一个新的缩放器实例
    features_scaler = MinMaxScaler(feature_range=(0, 1))
    df_train[features] = features_scaler.fit_transform(df_train[features])
    df_test[features] = features_scaler.transform(df_test[features])

    # 为目标变量创建一个新的缩放器实例
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    df_train['Lane 1 Flow (Veh/5 Minutes)'] = target_scaler.fit_transform(df_train[['Lane 1 Flow (Veh/5 Minutes)']])
    df_test['Lane 1 Flow (Veh/5 Minutes)'] = target_scaler.transform(df_test[['Lane 1 Flow (Veh/5 Minutes)']])

    # 使用缩放过的特征和目标变量创建数据集
    X_train, y_train = create_dataset(df_train[features].values, lags, df_train['Lane 1 Flow (Veh/5 Minutes)'].values)
    X_test, y_test = create_dataset(df_test[features].values, lags, df_test['Lane 1 Flow (Veh/5 Minutes)'].values)

    return X_train, y_train, X_test, y_test, features_scaler, target_scaler

if __name__ == "__main__":
    process_and_check_data()


