import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

sys.path.append('/root/method')
from model.model import get_lstm, get_gru, get_saes
from data.data import process_data, process_and_check_data

# 禁用警告
warnings.filterwarnings('ignore')

def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))

def explained_variance_score(y_true, y_pred):
    numerator = K.sum(K.square(y_true - K.mean(y_true)) - K.square(y_true - y_pred))
    denominator = K.sum(K.square(y_true - K.mean(y_true)))
    return numerator / (denominator + K.epsilon())

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def train_model(model, X_train, y_train, name, config):
    filepath = 'model/' + name + '_best.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    
    callbacks_list = [checkpoint, early_stopping]
    
    history = model.fit(
        X_train, y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_split=0.05,
        callbacks=callbacks_list
    )

    model.save('model/' + name + '.h5')
    return history

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

X_train, y_train, X_test, y_test, scaler = process_data(config["train_file"], config["test_file"], config["lag"])

process_and_check_data()

model_name = 'lstm'  # 或 'gru', 'saes' 根据需要选择

# 确保X_train是三维的，适合LSTM模型
if model_name in ['lstm', 'gru']:
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model_config = config["model_configs"][model_name]
model_config[0] = X_train.shape[1]  # 设置时间步长为输入形状的第二维

if model_name == 'lstm':
    model = get_lstm(model_config)
elif model_name == 'gru':
    model = get_gru(model_config)
elif model_name == 'saes':
    models = get_saes(model_config)
    history = train_saes(models, X_train, y_train, model_name, config)
else:
    raise ValueError(f"Unknown model type: {model_name}")

if model_name in ['lstm', 'gru']:
    history = train_model(model, X_train, y_train, model_name, config)

# 绘制训练和验证的损失曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


