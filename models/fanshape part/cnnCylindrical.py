import numpy as np
import glob
import os
import re
import tifffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
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

# 获取图像路径
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

# 定义极坐标转换函数
def cartesian_to_polar(image):
    """
    将图像从笛卡尔坐标转换为极坐标。
    """
    height, width = image.shape
    center_x, center_y = width // 2, height // 2
    y, x = np.indices((height, width))
    x = x - center_x
    y = y - center_y
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    r = (r / r.max()) * (height - 1)
    theta = (theta + np.pi) / (2 * np.pi) * (width - 1)
    polar_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            polar_image[i, j] = image[int(r[i, j]), int(theta[i, j])]
    return polar_image

# 将X中的每个图像转换为极坐标
X = np.stack([emcal_data, hcal_data, trkn_data, trkp_data], axis=-1)
Y = truth_data

for i in range(X.shape[0]):
    for j in range(X.shape[-1]):
        X[i, :, :, j] = cartesian_to_polar(X[i, :, :, j])

for i in range(Y.shape[0]):
    Y[i, :, :] = cartesian_to_polar(Y[i, :, :])

# 数据归一化
scalers_X = [MinMaxScaler() for _ in range(X.shape[-1])]
scaler_Y = MinMaxScaler()

for i in range(X.shape[-1]):
    X_channel = X[..., i].reshape(-1, 1)
    scalers_X[i].fit(X_channel)
    X[..., i] = scalers_X[i].transform(X_channel).reshape(X[..., i].shape)

Y = Y.reshape(-1, 1)
scaler_Y.fit(Y)
Y = scaler_Y.transform(Y)
Y = Y.reshape(-1, 56, 56, 1)  # 确保 Y 的形状为 (样本数, 高度, 宽度, 通道数)

# 检查 X 和 Y 的形状
print("Number of samples in X:", X.shape[0])
print("Number of samples in Y:", Y.shape[0])

# 划分数据集
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.33, random_state=42)

# 定义圆柱形卷积层
class CylindricalConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(CylindricalConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = 'SAME'  # 使用大写

    def build(self, input_shape):
        kernel_height, kernel_width = self.kernel_size
        self.kernel = self.add_weight(
            shape=(kernel_height, kernel_width, input_shape[-1], self.filters),
            initializer='glorot_uniform',
            trainable=True
        )

        # 创建圆柱形掩膜
        self.cylindrical_mask = self.create_cylindrical_mask(kernel_height, kernel_width)
    
    def create_cylindrical_mask(self, height, width):
        center_x, center_y = width // 2, height // 2
        y, x = np.indices((height, width))
        x = x - center_x
        y = y - center_y
        r = np.sqrt(x**2 + y**2)
        
        # 创建圆柱形掩膜
        radius_limit = min(center_x, center_y)
        cylindrical_mask = r <= radius_limit

        cylindrical_mask = cylindrical_mask.astype(np.float32)
        return tf.constant(cylindrical_mask, dtype=tf.float32)

    def call(self, inputs):
        kernel_height, kernel_width = self.kernel_size

        # 使用圆柱形掩膜对卷积核进行遮罩
        masked_kernel = self.kernel * tf.reshape(self.cylindrical_mask, (kernel_height, kernel_width, -1, 1))

        # 使用 tf.image.extract_patches 提取补丁
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, kernel_height, kernel_width, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding=self.padding
        )

        patches_shape = tf.shape(patches)
        num_patches = patches_shape[1] * patches_shape[2]
        patch_size = kernel_height * kernel_width * tf.shape(inputs)[-1]

        # 将卷积核应用于补丁
        patches = tf.reshape(patches, (-1, num_patches, patch_size))
        conv_out = tf.matmul(patches, tf.reshape(masked_kernel, (-1, self.filters)))
        conv_out = tf.reshape(conv_out, (tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], self.filters))

        return conv_out

# 定义模型
model = Sequential([
    Input(shape=(56, 56, 4)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    CylindricalConv2D(filters=64, kernel_size=(5, 5)),  # 使用圆柱形卷积掩膜
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(1, (1, 1), activation='linear', padding='same'),
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 打印模型结构
model.summary()

# 训练模型
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val))

# 保存模型
model.save('my_model1.keras')

# 使用测试集进行测试
y_pred = model.predict(X_test)

# 保存预测结果和真实值
np.save(os.path.join(save_path, 'predictions.npy'), y_pred)
np.save(os.path.join(save_path, 'truth.npy'), Y_test)
