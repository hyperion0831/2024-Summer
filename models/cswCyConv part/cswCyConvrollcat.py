import numpy as np
import glob
import os
import re
import tifffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
X = np.stack([emcal_data, hcal_data, trkn_data, trkp_data], axis=-1)
Y = truth_data

# 数据归一化
scalers_X = [MinMaxScaler() for _ in range(X.shape[-1])]
scaler_Y = MinMaxScaler()

# 记录Y的原始形状
Y_original_shape = Y.shape

# 对X的每个通道进行独立标准化
for i in range(X.shape[-1]):
    X_channel = X[..., i].reshape(-1, 1)
    scalers_X[i].fit(X_channel)
    X[..., i] = scalers_X[i].transform(X_channel).reshape(X[..., i].shape)

# 对Y进行标准化
Y = Y.reshape(-1, 1)
scaler_Y.fit(Y)
Y = scaler_Y.transform(Y)
Y = Y.reshape(Y_original_shape)

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
for i in range(X.shape[0]):
    for j in range(X.shape[-1]):
        X[i, :, :, j] = cartesian_to_polar(X[i, :, :, j])

# 划分训练集、验证集和测试集
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.33, random_state=42)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
Y_val = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)

# 自定义圆柱卷积层 (CyConv2d) with CSW mechanism
class CyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CyConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # 使用 torch.roll 对输入进行周期性扩展
        pad = self.kernel_size // 2
        rolled_x = torch.roll(x, shifts=pad, dims=2)  # 在宽度维度上滚动
        rolled_x = torch.roll(rolled_x, shifts=pad, dims=3)  # 在高度维度上滚动

        # 进行卷积操作
        conv_out = F.conv2d(rolled_x, self.weight, stride=self.stride, padding=self.padding)
        
        # 使用 torch.cat 将卷积结果沿通道维度拼接在一起
        combined_out = torch.cat([conv_out, conv_out], dim=1)  # 这里假设拼接两次卷积结果

        return combined_out


# 定义模型，替换Conv2D为CyConv2d
class CyCNN(nn.Module):
    def __init__(self):
        super(CyCNN, self).__init__()
        self.layer1 = CyConv2d(4, 32, 3, padding=1)
        self.layer2 = CyConv2d(64, 32, 3, padding=1)
        self.layer3 = CyConv2d(64, 64, 5, padding=2)
        self.layer4 = CyConv2d(128, 32, 3, padding=1)
        self.layer5 = CyConv2d(64, 1, 1)
        self.final_conv = nn.Conv2d(2, 1, kernel_size=1)  # 压缩到一个通道

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        x = self.final_conv(x)  # 使用 final_conv 调整通道数
        return x


# 创建模型
model = CyCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练模型
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = F.interpolate(targets, size=outputs.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                targets = F.interpolate(targets, size=outputs.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.6f}')

# 创建数据加载器
train_loader = DataLoader(list(zip(X_train, Y_train)), batch_size=32, shuffle=True)
val_loader = DataLoader(list(zip(X_val, Y_val)), batch_size=32, shuffle=False)

# 训练模型
train_model(model, train_loader, val_loader, optimizer, criterion)

# 使用测试集进行测试
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for inputs, targets in DataLoader(list(zip(X_test, Y_test)), batch_size=32):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        targets = F.interpolate(targets, size=outputs.shape[2:], mode='bilinear', align_corners=False)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# 保存预测结果和真实值
np.save(os.path.join(save_path, 'predictions.npy'), all_preds)
np.save(os.path.join(save_path, 'truth.npy'), all_targets)

# 可视化训练过程
plt.figure()
plt.plot(history.history['loss'], label='训练集损失')
plt.plot(history.history['val_loss'], label='验证集损失')
plt.title('模型损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()
plt.show()
