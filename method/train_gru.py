import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from model.model import get_gru
from data.data import process_data

# 定义评估函数
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def r_squared(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

# 训练GRU模型的函数
def train_gru(X_train, y_train, config, model_name='gru'):
    # 获取模型配置并创建模型
    model_config = config["model_configs"][model_name]
    model_config[0] = X_train.shape[1]  # 设置时间步长为输入形状的第二维
    model = get_gru(model_config)

    # 设置模型保存路径和回调函数
    filepath = f'model/{model_name}_best.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1)

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
    model.save(f'model/{model_name}.h5')
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

# 主函数
def main():
    # 配置参数
    config = {
        "batch_size": 128,
        "epochs": 900,
        "lag": 12,
        "train_file": 'data/train.csv',
        "test_file": 'data/test.csv',
        "model_configs": {
            "gru": [None, 64, 64, 1]
        }
    }

    # 数据处理
    X_train, y_train, _, _, _ = process_data(config["train_file"], config["test_file"], config["lag"])

    # 确保X_train是三维的，适合GRU模型
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # 训练模型
    history = train_gru(X_train, y_train, config)

    # 绘制训练和验证损失曲线
    plot_loss(history)

    print("GRU模型训练完成")

if __name__ == '__main__':
    main()