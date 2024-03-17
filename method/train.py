import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

sys.path.append('/root/method')
from model.model import get_gru, get_lstm, get_saes
from data.data import process_data, process_saes_data

# 定义评估函数
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))

def explained_variance_score(y_true, y_pred):
    numerator = K.sum(K.square(y_true - K.mean(y_true)) - K.square(y_true - y_pred))
    denominator = K.sum(K.square(y_true - K.mean(y_true)))
    return numerator / (denominator + K.epsilon())

def flatten_input_data(X):
    """将输入数据从三维 (samples, timesteps, features) 展平为二维 (samples, timesteps*features)"""
    return X.reshape((X.shape[0], np.prod(X.shape[1:])))

def train_model(model, X_train, y_train, config, model_type='gru'):
    # 根据模型类型设置模型保存路径和回调函数
    filepath = f'model/{model_type}_best.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=30 if model_type != 'lstm' else 100, verbose=1)

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mse', rmse, 'mae', r_squared])

    # 训练模型
    history = model.fit(
        X_train, y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_split=0.05,
        callbacks=[checkpoint, early_stopping]
    )

    # 保存训练好的模型
    model.save(f'model/{model_type}.h5')
    return history


# 绘制训练和验证的损失曲线
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    # 通用配置参数
    config = {
        "batch_size": 128,
        "epochs": 900,
        "lag": 12,
        "train_file": 'data/train.csv',
        "test_file": 'data/test.csv',
        "model_configs": {
            "lstm": [None, 64, 64, 1],
            "gru": [None, 64, 64, 1],
            "saes": [None, 400, 400, 400, 1]
        }
    }

    model_name = 'gru'  # 可以更改为 'lstm' 或 'saes'

    # 数据处理
    X_train, y_train, X_test, y_test, scaler = process_data(config["train_file"], config["test_file"], config["lag"])
    
    if model_name in ['lstm', 'gru']:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    elif model_name == 'saes':
        X_train = X_train.reshape((X_train.shape[0], -1))

    # 训练模型
    if model_name == 'saes':
        models = get_saes(config["model_configs"][model_name])
        history = train_model(models, X_train, y_train, config, model_name)
    else:
        model_config = config["model_configs"][model_name]
        model_config[0] = X_train.shape[1]  # 设置时间步长为输入形状的第二维
        if model_name == 'lstm':
            model = get_lstm(model_config)
        elif model_name == 'gru':
            model = get_gru(model_config)
        history = train_model(model, X_train, y_train, config, model_name)

    # 绘制训练和验证损失曲线
    plot_loss(history)
    print(f"{model_name.upper()}模型训练完成")

if __name__ == '__main__':
    main()