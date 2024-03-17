import pandas as pd
from datetime import datetime, timedelta

# 设置header
new_headers = [
    'Timestamp', 'Station', 'District', 'Freeway', 'Direction',
    'Lane Type', 'Station Length', 'Samples', '% Observed',
    'Total Flow', 'Avg Occupancy', 'Avg Speed',
    'Lane 1 Samples', 'Lane 1 Flow', 'Lane 1 Avg Occ',
    'Lane 1 Avg Speed', 'Lane 1 Observed'
]
# 文件前缀和日期范围
file_prefix = 'd12_text_station_5min_'
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 1, 2)
​
# 创建一个空DataFrame来累积所有日期的数据
accumulated_data = pd.DataFrame()
​
# 生成日期列表
date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
​
for date in date_list:
    file_suffix = date.strftime('%Y_%m_%d')
    file_path = f"{file_prefix}{file_suffix}.txt.gz"  # 根据实际存储位置可能需要调整路径
try:
    # 读取每日的文件
    daily_data = pd.read_csv(
        file_path,
        compression='gzip',
        header=None,
        names=new_headers,
        usecols=range(len(new_headers))
    )

    # 数据过滤并创建副本以避免警告
    filtered_data = daily_data[(daily_data['Lane 1 Observed'] == 1) & (daily_data['% Observed'] == 100)].copy()
    filtered_data['Timestamp'] = pd.to_datetime(filtered_data['Timestamp'])
    filtered_data['Lane 1 Flow'] = filtered_data['Lane 1 Flow'].astype(int)
​
# 累积到总DataFrame
accumulated_data = pd.concat([accumulated_data, filtered_data], ignore_index=True)
​
except FileNotFoundError:
print(f"File not found: {file_path}")
except Exception as e:
print(f"Error processing file {file_path}: {e}")
​
# 按照Timestamp整合数据，确保完全相同的Timestamp才整合
final_data = accumulated_data.groupby('Timestamp').agg({
    'Lane 1 Flow': 'sum',
    'Lane 1 Observed': 'max',
    '% Observed': 'max'
}).reset_index()
​
# Timestamp格式转换
final_data['Timestamp'] = final_data['Timestamp'].dt.strftime('%m/%d/%Y %H:%M')
# 保存到CSV
final_data.to_csv('final_aggregated_data.csv', index=False)
import pandas as pd
from datetime import datetime, timedelta
​
# 初始化新的列名列表
new_headers = [
    'Timestamp', 'Station', 'District', 'Freeway', 'Direction',
    'Lane Type', 'Station Length', 'Samples', '% Observed',
    'Total Flow', 'Avg Occupancy', 'Avg Speed',
    'Lane 1 Samples', 'Lane 1 Flow', 'Lane 1 Avg Occ',
    'Lane 1 Avg Speed', 'Lane 1 Observed'
]
​
# 文件前缀和日期范围
file_prefix = 'd12_text_station_5min_'
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 1, 2)
​
# 创建一个空DataFrame来累积所有日期的数据
accumulated_data = pd.DataFrame()
​
# 生成日期列表
date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
​
for date in date_list:
    file_date = date.strftime('%Y_%m_%d')
    file_path = f"{file_prefix}{file_date}.txt.gz"  # 调整路径以匹配您的文件存储位置
​
try:
    daily_data = pd.read_csv(
        file_path,
        compression='gzip',
        header=None,
        names=new_headers,
        usecols=range(len(new_headers))
    )
    daily_data['Timestamp'] = pd.to_datetime(daily_data['Timestamp'])
    daily_data = daily_data[(daily_data['Lane 1 Observed'] == 1) & (daily_data['% Observed'] == 100)]
    daily_data['Lane 1 Flow'] = daily_data['Lane 1 Flow'].astype(int)
​
# 无需再次分组同日数据，直接累积到总DataFrame
accumulated_data = pd.concat([accumulated_data, daily_data], ignore_index=True)
​
except FileNotFoundError:
print(f"File not found: {file_path}")
except Exception as e:
print(f"Error processing file {file_path}: {e}")
​
# 对所有累积数据进行最终的Timestamp分组和整合
final_data = accumulated_data.groupby('Timestamp').agg({
    'Lane 1 Flow': 'sum',
    'Lane 1 Observed': 'max',
    '% Observed': 'max'
}).reset_index()
​
final_data['Timestamp'] = final_data['Timestamp'].dt.strftime('%m/%d/%Y %H:%M')
​
final_data.to_csv('final_aggregated_data.csv', index=False)
​
​

