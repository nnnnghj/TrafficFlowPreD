# 使用官方的Python 3.6镜像作为基础镜像
FROM python:3.6

# 安装TensorFlow GPU版本
RUN pip install tensorflow-gpu==1.5.0

# 安装Keras
RUN pip install keras==2.1.3

# 安装scikit-learn
RUN pip install scikit-learn==0.19

# 设置工作目录
WORKDIR /app

# 将当前目录下的所有文件复制到容器的工作目录中
COPY . /app

# 运行时容器监听的端口号
EXPOSE 8888
