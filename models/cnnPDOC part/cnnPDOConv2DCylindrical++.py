import numpy as np
import glob
import os
import re
import tifffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, UpSampling2D
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

# 定义极坐标转换函数
def cartesian_to_polar(image):
    """
    将输入图像从笛卡尔坐标转换为极坐标。
    输入图像的维度为 (batch_size, height, width, channels)。
    """
    batch_size, height, width, channels = image.shape
    
    # 创建极坐标的r和theta网格
    center_x, center_y = width // 2, height // 2
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    x = x - center_x
    y = y - center_y
    
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    
    # 归一化r和theta
    r = r / r.max()  
    theta = (theta + np.pi) / (2 * np.pi)  
    
    # 根据r和theta对图像进行重采样
    r_scaled = (r * (height - 1)).astype(np.int32)
    theta_scaled = (theta * (width - 1)).astype(np.int32)
    
    polar_image = np.zeros_like(image)
    for i in range(batch_size):
        for c in range(channels):
            polar_image[i, :, :, c] = image[i, r_scaled, theta_scaled, c]
    
    return polar_image

# 将数据转换为极坐标形式
X_train_polar = cartesian_to_polar(X_train)
X_val_polar = cartesian_to_polar(X_val)
X_test_polar = cartesian_to_polar(X_test)

# 定义基于PDOs的自定义圆柱形卷积层
class CylindricalPDOConv2D(Layer):
    def __init__(self, filters, **kwargs):
        super(CylindricalPDOConv2D, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.kernel_I = self.add_weight(shape=(3, 3, input_shape[-1], self.filters),
                                        initializer='glorot_uniform', trainable=True)
        self.kernel_dx = self.add_weight(shape=(3, 3, input_shape[-1], self.filters),
                                         initializer='glorot_uniform', trainable=True)
        self.kernel_dy = self.add_weight(shape=(3, 3, input_shape[-1], self.filters),
                                         initializer='glorot_uniform', trainable=True)
        self.kernel_dz = self.add_weight(shape=(3, 3, input_shape[-1], self.filters),
                                         initializer='glorot_uniform', trainable=True)
        self.kernel_lap = self.add_weight(shape=(3, 3, input_shape[-1], self.filters),
                                          initializer='glorot_uniform', trainable=True)
        super(CylindricalPDOConv2D, self).build(input_shape)

    def call(self, inputs):
        I = tf.identity(inputs)
        
        # x方向梯度
        dx = tf.image.sobel_edges(inputs)[..., 0]
        
        # y方向梯度
        dy = tf.image.sobel_edges(inputs)[..., 1]
        
        # z方向梯度
        dz_kernel = tf.constant([[0, 1, 0], [0, 0, 0], [0, -1, 0]], dtype=tf.float32)
        dz_kernel = tf.reshape(dz_kernel, (3, 3, 1, 1))
        dz_kernel = tf.tile(dz_kernel, [1, 1, tf.shape(inputs)[-1], self.filters])
        dz = tf.nn.conv2d(inputs, dz_kernel, strides=[1, 1, 1, 1], padding='SAME')

        # 拉普拉斯算子
        lap_kernel = tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=tf.float32)
        lap_kernel = tf.reshape(lap_kernel, (3, 3, 1, 1))
        lap_kernel = tf.tile(lap_kernel, [1, 1, tf.shape(inputs)[-1], self.filters])
        lap = tf.nn.conv2d(inputs, lap_kernel, strides=[1, 1, 1, 1], padding='SAME')

        # PDO卷积操作
        conv_I = tf.nn.conv2d(I, self.kernel_I, strides=[1, 1, 1, 1], padding='SAME')
        conv_dx = tf.nn.conv2d(dx, self.kernel_dx, strides=[1, 1, 1, 1], padding='SAME')
        conv_dy = tf.nn.conv2d(dy, self.kernel_dy, strides=[1, 1, 1, 1], padding='SAME')
        conv_dz = tf.nn.conv2d(dz, self.kernel_dz, strides=[1, 1, 1, 1], padding='SAME')
        conv_lap = tf.nn.conv2d(lap, self.kernel_lap, strides=[1, 1, 1, 1], padding='SAME')

        # 输出是所有卷积的叠加
        return conv_I + conv_dx + conv_dy + conv_dz + conv_lap

# 定义模型
input_layer = Input(shape=(56, 56, 4))
x = CylindricalPDOConv2D(filters=64)(input_layer)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same')(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same')(x)
x = tf.keras.layers.ReLU()(x)
x = UpSampling2D(size=(2, 2))(x)  # 添加上采样层，将输出形状调整为 (56, 56, 1)
output_layer = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 训练模型
history = model.fit(X_train_polar, Y_train, validation_data=(X_val_polar, Y_val), epochs=100, batch_size=16)

# 评估模型
test_loss = model.evaluate(X_test_polar, Y_test)

# 保存预测结果
predictions = model.predict(X_test_polar)
np.save(os.path.join(save_path, 'predictions.npy'), predictions)
np.save(os.path.join(save_path, 'truth.npy'), Y_test)

# 打印测试损失
print("Test loss:", test_loss)