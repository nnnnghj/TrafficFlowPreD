import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from data.data import process_data
from sklearn.metrics import mean_absolute_error, explained_variance_score

# 自定义的评估函数
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def r_squared(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = np.where(y_true != 0)  # 找出实际值非零的索引
    y_true_non_zero = y_true[non_zero]
    y_pred_non_zero = y_pred[non_zero]
    return np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100

# 加载模型
model = load_model('model/gru_best.h5', custom_objects={'rmse': rmse, 'r_squared': r_squared})

# 假设process_data函数已经被定义并可以正常使用
_, _, X_test, y_test, scaler = process_data('data/train.csv', 'data/test.csv', 12)

# 确保X_test是三维的，适合GRU模型
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 进行预测
y_pred = model.predict(X_test)

# 将预测和实际标签的缩放逆转（如果使用了缩放）
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# 评估性能
mse = tf.keras.metrics.mean_squared_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten()).numpy()
rmse_value = rmse(tf.constant(y_test_rescaled.flatten(), dtype=tf.float32), tf.constant(y_pred_rescaled.flatten(), dtype=tf.float32)).numpy()
r2_value = r_squared(tf.constant(y_test_rescaled.flatten(), dtype=tf.float32), tf.constant(y_pred_rescaled.flatten(), dtype=tf.float32)).numpy()
mae_value = mean_absolute_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten())
explained_variance = explained_variance_score(y_test_rescaled.flatten(), y_pred_rescaled.flatten())
mape_value = mape(y_test_rescaled.flatten(), y_pred_rescaled.flatten())

print(f"测试集 MSE: {mse}")
print(f"测试集 RMSE: {rmse_value}")
print(f"测试集 R-squared: {r2_value}")
print(f"测试集 MAE: {mae_value}")
print(f"测试集 Explained Variance Score: {explained_variance}")
print(f"测试集 MAPE: {mape_value} %")

