import numpy as np
import glob
import os
import re
import tifffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# 数据文件夹路径
folder_path = 'C:/Users/25610/Desktop/Online/Gauss_S1.00_NL0.30_B0.50'
# 保存路径
save_path = 'C:/Users/25610/Desktop/online3/save3'
os.makedirs(save_path, exist_ok=True)

# 自然排序函数
def natural_sort_key(s):
    sub_strings = re.split(r'(\d+)', s)
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    return sub_strings

# 读取图像数据
def load_images(file_list):
    images = [tiff.imread(p) for p in file_list]
    return np.array(images)

# 获取图像路径
emcal_list = sorted(glob.glob(os.path.join(folder_path, 'emcal_*')), key=natural_sort_key)
hcal_list = sorted(glob.glob(os.path.join(folder_path, 'hcal_*')), key=natural_sort_key)
trkn_list = sorted(glob.glob(os.path.join(folder_path, 'trkn_*')), key=natural_sort_key)
trkp_list = sorted(glob.glob(os.path.join(folder_path, 'trkp_*')), key=natural_sort_key)
truth_list = sorted(glob.glob(os.path.join(folder_path, 'truth_*')), key=natural_sort_key)

# 加载数据
emcal_data = load_images(emcal_list)
hcal_data = load_images(hcal_list)
trkn_data = load_images(trkn_list)
trkp_data = load_images(trkp_list)
truth_data = load_images(truth_list)

# 数据归一化
scalers_X = [MinMaxScaler() for _ in range(4)]
scaler_Y = MinMaxScaler()

X = np.stack([emcal_data, hcal_data, trkn_data, trkp_data], axis=-1)
Y = truth_data

Y_original_shape = Y.shape

for i in range(X.shape[-1]):
    X_channel = X[..., i].reshape(-1, 1)
    scalers_X[i].fit(X_channel)
    X[..., i] = scalers_X[i].transform(X_channel).reshape(X[..., i].shape)

Y = Y.reshape(-1, 1)
scaler_Y.fit(Y)
Y = scaler_Y.transform(Y)
Y = Y.reshape(Y_original_shape)

# 划分数据集
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.33, random_state=42)

# 定义基于PDOs的自定义卷积层
class PDOConv2D(Layer):
    def __init__(self, filters, **kwargs):
        super(PDOConv2D, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.kernel_I = self.add_weight(shape=(3, 3, input_shape[-1], self.filters),
                                        initializer='glorot_uniform', trainable=True)
        self.kernel_dx = self.add_weight(shape=(3, 3, input_shape[-1], self.filters),
                                         initializer='glorot_uniform', trainable=True)
        self.kernel_dy = self.add_weight(shape=(3, 3, input_shape[-1], self.filters),
                                         initializer='glorot_uniform', trainable=True)
        self.kernel_lap = self.add_weight(shape=(3, 3, input_shape[-1], self.filters),
                                          initializer='glorot_uniform', trainable=True)
        super(PDOConv2D, self).build(input_shape)

    def call(self, inputs):
        I = tf.identity(inputs)
        dx = tf.image.sobel_edges(inputs)[..., 0]  # x方向的梯度
        dy = tf.image.sobel_edges(inputs)[..., 1]  # y方向的梯度
        
        # 使用拉普拉斯卷积核计算近似拉普拉斯
        lap_kernel = tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=tf.float32)
        lap_kernel = tf.reshape(lap_kernel, (3, 3, 1, 1))  # 确保形状是 (3, 3, 1, 1)
        lap_kernel = tf.tile(lap_kernel, [1, 1, tf.shape(inputs)[-1], self.filters])  # 扩展到正确的通道数
        lap = tf.nn.conv2d(inputs, lap_kernel, strides=[1, 1, 1, 1], padding='SAME')

        # 使用PDOs进行卷积操作
        conv_I = tf.nn.conv2d(I, self.kernel_I, strides=[1, 1, 1, 1], padding='SAME')
        conv_dx = tf.nn.conv2d(dx, self.kernel_dx, strides=[1, 1, 1, 1], padding='SAME')
        conv_dy = tf.nn.conv2d(dy, self.kernel_dy, strides=[1, 1, 1, 1], padding='SAME')
        conv_lap = tf.nn.conv2d(lap, self.kernel_lap, strides=[1, 1, 1, 1], padding='SAME')

        return conv_I + conv_dx + conv_dy + conv_lap

# 定义模型
input_tensor = Input(shape=(56, 56, 4))
x = PDOConv2D(filters=32)(input_tensor)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
output_tensor = tf.keras.layers.Conv2D(1, (1, 1), activation='linear', padding='same')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)

# 编译并训练模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.summary()

history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val))

# 保存模型和预测结果
model.save('my_model1.keras')
y_pred = model.predict(X_test)
np.save(os.path.join(save_path, 'predictions.npy'), y_pred)
np.save(os.path.join(save_path, 'truth.npy'), Y_test)
