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

# 定义改进的极坐标转换函数，使用双线性插值
def cartesian_to_polar(image):
    """
    使用双线性插值将图像从笛卡尔坐标转换为极坐标。
    """
    height, width = image.shape
    center_x, center_y = width // 2, height // 2
    max_radius = np.sqrt(center_x**2 + center_y**2)

    # 创建空的极坐标图像
    polar_image = np.zeros_like(image)

    # 创建极坐标的r和theta网格
    theta, radius = np.meshgrid(np.linspace(0, 2 * np.pi, width), np.linspace(0, max_radius, height))

    # 计算笛卡尔坐标
    x = radius * np.cos(theta) + center_x
    y = radius * np.sin(theta) + center_y

    # 双线性插值
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)

    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    polar_image = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return polar_image

# 将X中的每个图像转换为极坐标
for i in range(X.shape[0]):
    for j in range(X.shape[-1]):
        X[i, :, :, j] = cartesian_to_polar(X[i, :, :, j])

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
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=32)

# 评估模型
test_loss = model.evaluate(X_test, Y_test)

# 保存预测结果
predictions = model.predict(X_test)
np.save(os.path.join(save_path, 'predictions.npy'), predictions)
np.save(os.path.join(save_path, 'truth.npy'), Y_test)

# 打印测试损失
print("Test loss:", test_loss)