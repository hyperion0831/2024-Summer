import numpy as np
import glob
import os
import re
import tifffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.optimizers import Adam

# 指定数据文件夹路径
folder_path = 'C:/Users/25610/Desktop/Online/Gauss_S1.00_NL0.30_B0.50'
# 指定保存路径
save_path = 'C:/Users/25610/Desktop/online3/save3'
# 确保目录存在
os.makedirs(save_path, exist_ok=True)

# 定义自然排序函数
def natural_sort_key(s):
    sub_strings = re.split(r'(\d+)', s)
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    return sub_strings

# 获取不同类别图片路径并加入到对应的列表中
emcal_list = sorted(glob.glob(os.path.join(folder_path, 'emcal_*')), key=natural_sort_key)
hcal_list = sorted(glob.glob(os.path.join(folder_path, 'hcal_*')), key=natural_sort_key)
trkn_list = sorted(glob.glob(os.path.join(folder_path, 'trkn_*')), key=natural_sort_key)
trkp_list = sorted(glob.glob(os.path.join(folder_path, 'trkp_*')), key=natural_sort_key)
truth_list = sorted(glob.glob(os.path.join(folder_path, 'truth_*')), key=natural_sort_key)

# 读取图像数据
def load_images(file_list):
    images = [tiff.imread(p) for p in file_list]
    return np.array(images)

emcal_data = load_images(emcal_list)
hcal_data = load_images(hcal_list)
trkn_data = load_images(trkn_list)
trkp_data = load_images(trkp_list)
truth_data = load_images(truth_list)

# 合并前四张图作为输入，第五张图作为输出
# 计算四个通道的均值以生成一个通道的数据
X = np.mean(np.stack([emcal_data, hcal_data, trkn_data, trkp_data], axis=-1), axis=-1, keepdims=True)

# 确保 X 的形状为 (num_samples, 56, 56, 1, 1)
X = np.expand_dims(X, axis=-1)  # Shape: (num_samples, 56, 56, 1, 1)

Y = truth_data

# 数据归一化
# 对X进行标准化
X = X.reshape(-1, 1)  # 将数据展平以便标准化
scaler_X = MinMaxScaler()
scaler_X.fit(X)
X = scaler_X.transform(X)
X = X.reshape(-1, 56, 56, 1, 1)  # 还原形状

# 对Y进行标准化
Y = Y.reshape(-1, 1)
scaler_Y = MinMaxScaler()
scaler_Y.fit(Y)
Y = scaler_Y.transform(Y)
Y = Y.reshape(truth_data.shape)

# 划分训练集、验证集和测试集
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.33, random_state=42)

# 确保Y的形状与模型输出形状相匹配
Y_train = np.expand_dims(Y_train, axis=-1)
Y_val = np.expand_dims(Y_val, axis=-1)
Y_test = np.expand_dims(Y_test, axis=-1)

# 打印数据形状以进行调试
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of Y_train: {Y_train.shape}")
print(f"Shape of X_val: {X_val.shape}")
print(f"Shape of Y_val: {Y_val.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of Y_test: {Y_test.shape}")

# 定义3D卷积自编码器模型
model = Sequential([
    Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=(56, 56, 1, 1)),
    Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
    Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
    Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
    Conv3D(1, (1, 1, 1), activation='linear', padding='same')
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 打印模型结构，检查输出形状是否正确
model.summary()

# 训练模型
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val))

# 保存模型
model.save('my_model1.keras')

# 使用测试集进行测试
y_pred = model.predict(X_test)

# 打印预测和真实标签的维度
print(f"Shape of y_pred: {y_pred.shape}")
print(f"Shape of Y_test: {Y_test.shape}")

# 确保预测结果的形状与Y_test相同
y_pred = np.squeeze(y_pred, axis=-1)  # 去掉最后一个维度
Y_test = np.squeeze(Y_test, axis=-1)  # 去掉最后一个维度

# 打印去除维度后的形状
print(f"Shape of squeezed y_pred: {y_pred.shape}")
print(f"Shape of squeezed Y_test: {Y_test.shape}")

# 确保维度匹配
assert y_pred.shape == Y_test.shape, "预测结果和真实标签的形状不匹配"

# 保存预测结果和真实值
np.save(os.path.join(save_path, 'predictions.npy'), y_pred)
np.save(os.path.join(save_path, 'truth.npy'), Y_test)

# 可视化训练过程
plt.figure()
plt.plot(history.history['loss'], label='训练集损失')
plt.plot(history.history['val_loss'], label='验证集损失')
plt.title('模型损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()
plt.show()
