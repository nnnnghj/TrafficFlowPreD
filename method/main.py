import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from data.data import process_data, process_saes_data
from sklearn.preprocessing import MinMaxScaler

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    y_true_non_zero = y_true[non_zero]
    y_pred_non_zero = y_pred[non_zero]
    return np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100

def r_squared(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

def eva_regress(y_true, y_pred):
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print(f'explained_variance_score: {vs}')
    print(f'mape: {MAPE(y_true, y_pred)}%')
    print(f'mae: {mae}')
    print(f'mse: {mse}')
    print(f'rmse: {np.sqrt(mse)}')
    print(f'r2: {r2}')

# 绘图函数，传入时间序列作为x轴
def plot_results(time_series, y_true, y_preds, names, selected_names=None):
    plt.figure(figsize=(15, 5))
    num_points_to_plot = len(y_true) // 5  # 取20%的数据点用于绘图
    time_series = time_series[:num_points_to_plot]
    y_true = y_true[:num_points_to_plot]
    plt.plot(time_series, y_true, label='True Data', color='black', linestyle='--')
    
    if selected_names is None:
        selected_names = names
        
    for name, y_pred in zip(names, y_preds):
        if name in selected_names:
            y_pred = y_pred[:num_points_to_plot]
            plt.plot(time_series, y_pred, label=name)
    
    plt.title('Traffic Flow Prediction')
    plt.xlabel('Time')
    plt.ylabel('Flow')
    plt.gcf().autofmt_xdate()  # 自动格式化日期显示
    plt.legend()
    plt.show()

def plot_selected_results(y_true, y_preds, selected_names):
    plt.figure(figsize=(15, 5))
    plt.plot(y_true, label='True Data', color='black', linestyle='--')
    colors = ['blue', 'green']
    for name, y_pred, color in zip(selected_names, y_preds, colors):
        if name in selected_names:
            plt.plot(y_pred, label=name, color=color)
    plt.title('Traffic Flow Prediction (Selected Models)')
    plt.xlabel('Time')
    plt.ylabel('Flow')
    plt.legend()
    plt.show()
    
def main():
    config = {
        "batch_size": 128,
        "epochs": 900,
        "lag": 12,
        "train_file": 'data/train_saes.csv',
        "test_file": 'data/test_saes.csv',
        "freq": '5T',
        "model_configs": {
            "saes": [96, 400, 400, 400, 1]
        }
    }

    y_preds = []
    models = ['lstm', 'gru', 'saes']
    model_filenames = {
        'lstm': 'lstm_best.h5',
        'gru': 'gru_best.h5',
        'saes': 'saes_best_model.h5'
    }
    
    start_time_str = "01/26/2024 14:20"
    interval_minutes = 5  # The interval in the data, assumed to be 5 minutes
    
    for model_name in models:
        model_path = f'model/{model_filenames[model_name]}'
        model = load_model(model_path, custom_objects={'rmse': rmse, 'r_squared': r_squared})

        if model_name == 'saes':
            X_train_saes, y_train_saes, X_test_saes, y_test_saes, features_scaler_saes, target_scaler_saes = process_saes_data(
                config["train_file"], config["test_file"], config["lag"]
            )
            X_test_saes_flat = X_test_saes.reshape((X_test_saes.shape[0], -1))
            y_pred_saes = model.predict(X_test_saes_flat)
            y_pred_saes_rescaled = target_scaler_saes.inverse_transform(y_pred_saes)
            pd.DataFrame(y_pred_saes_rescaled.flatten()).to_csv(f'data/pred_{model_name}.csv', index=False, header=[f"{model_name.upper()}_Prediction"])
            print(f"SAEs模型预测结果已保存到CSV文件。")
            y_preds.append(y_pred_saes_rescaled.flatten())
            print(f'Model: {model_name}')
            eva_regress(y_test_saes.flatten(), y_pred_saes_rescaled.flatten())
        else:
            # 对于非SAEs模型，这里的数据处理可能需要根据实际情况进行调整
            X_train, y_train, X_test, y_test, scaler = process_data(config["train_file"], config["test_file"], config["lag"])
            X_test_transformed = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            y_pred = model.predict(X_test_transformed)
            y_pred_rescaled = scaler.inverse_transform(y_pred)
            y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
            pd.DataFrame(y_pred_rescaled.flatten()).to_csv(f'data/pred_{model_name}.csv', index=False, header=[f"{model_name.upper()}_Prediction"])
            y_preds.append(y_pred_rescaled.flatten())
            print(f'Model: {model_name}')
            eva_regress(y_test_rescaled.flatten(), y_pred_rescaled.flatten())

        # Assuming the start time of the test set is known and is 01/26/2024 14:20
    

    # Generate the time series for the x-axis in the plot
    start_time = pd.to_datetime(start_time_str)
    time_series = [start_time + pd.Timedelta(minutes=i * interval_minutes) for i in range(len(y_test_rescaled.flatten()))]
    
    def save_predictions_with_timestamps(predictions, timestamps, filename):
        predictions_with_timestamps = pd.DataFrame({
            'Timestamp': timestamps,
            'Prediction': predictions
        })
        predictions_with_timestamps.to_csv(filename, index=False)
        
    timestamps = [start_time + pd.Timedelta(minutes=i*interval_minutes) for i in range(len(y_test_rescaled))]
    
    for i, model_name in enumerate(models):
        save_predictions_with_timestamps(y_preds[i], timestamps, f'data/pred_{model_name}.csv')

    # Select only 20% of the test set data for plotting
    num_points_to_plot = len(y_test_rescaled.flatten()) // 5
    y_true = y_test_rescaled.flatten()[:num_points_to_plot]
    time_series = time_series[:num_points_to_plot]

    # Plot results for all models
    plot_results(time_series, y_true, [pred[:num_points_to_plot] for pred in y_preds], models)

    # Plot results for selected models
    selected_models = ['lstm', 'gru']
    selected_preds = [y_preds[models.index(m)][:num_points_to_plot] for m in selected_models]
    plot_results(time_series, y_true, selected_preds, selected_models)

# The plot_results function would look something like this:
def plot_results(time_series, y_true, y_preds, names, selected_names=None):
    plt.figure(figsize=(15, 5))
    plt.plot(time_series, y_true, label='True Data', color='black', linestyle='--')
    colors = ['blue', 'green', 'red', 'orange']
    for name, y_pred, color in zip(names, y_preds, colors):
        plt.plot(time_series, y_pred, label=name, color=color)
    plt.title('Traffic Flow Prediction')
    plt.xlabel('Time')
    plt.ylabel('Flow')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    plt.show()

if __name__ == '__main__':
    main()
