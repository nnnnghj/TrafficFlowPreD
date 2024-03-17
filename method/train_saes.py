import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from model.model import get_saes
from data.data import process_saes_data
import matplotlib.pyplot as plt

# 自定义的评估函数
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def r_squared(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

def flatten_input_data(X):
    """将输入数据从三维 (samples, timesteps, features) 展平为二维 (samples, timesteps*features)"""
    return X.reshape((X.shape[0], np.prod(X.shape[1:])))

# 训练SAEs模型的函数
def train_saes(X_train, y_train, config, model_name='saes'):
    # 获取模型配置并创建模型
    model_config = config["model_configs"][model_name]
    model_config[0] = X_train.shape[1]  # 设置时间步长为输入形状的第二维
    models = get_saes(model_config)

    for i, model in enumerate(models):
        # 设置模型保存路径和回调函数
        filepath = f'model/{model_name}_model_{i}_best.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)

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
        model.save(f'model/{model_name}_model_{i}.h5')


    return models

def main():
    config = {
        "batch_size": 128,
        "epochs": 900,
        "lag": 12,
        "train_file": 'data/train_saes.csv',
        "test_file": 'data/test_saes.csv',
        "model_configs": {
        "saes": [96, 400, 400, 400, 1]  # 假设处理后的数据维度是 96
        }
    }

    try:
        # 调用 process_saes_data 并接收返回的所有值，包括特征缩放器和目标缩放器
        X_train_saes, y_train_saes, X_test_saes, y_test_saes, features_scaler_saes, target_scaler_saes = process_saes_data(
            config["train_file"], config["test_file"], config["lag"]
        )
        if X_train_saes is None or X_test_saes is None or y_train_saes is None or y_test_saes is None:
            print("数据处理失败，一些数据为空，请检查文件路径和名称是否正确")
            return
        else:
            print(f"数据处理成功，训练数据形状: {X_train_saes.shape}, 训练标签形状: {y_train_saes.shape}")
    except Exception as e:
        print(f"处理SAEs数据时出错: {e}")
        return


    input_dim = X_train_saes.shape[1] * X_train_saes.shape[2]
    models = get_saes(input_dim)

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
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()