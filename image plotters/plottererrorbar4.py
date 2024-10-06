import numpy as np
import matplotlib.pyplot as plt
import os

# 指定保存路径
save_path1 = 'C:/Users/25610/Desktop/online3/savenewPDOCcylindrical'
save_path2 = 'C:/Users/25610/Desktop/online3/savenewPDOCcylindrical+'
save_path3 = 'C:/Users/25610/Desktop/online3/savenewPDOCcylindrical++'
save_path4 = 'C:/Users/25610/Desktop/online3/savenewPDOCcylindrical++Bilinear'  # 新增路径

# 初始化
std_devs1 = {}
std_devs2 = {}
std_devs3 = {}
std_devs4 = {}  # 新增字典

def calculate_std(y_pred, Y_test):
    y_pred_flat = y_pred.flatten()
    Y_test_flat = Y_test.flatten()
    
    # 排除真实值为0的情况
    non_zero_mask = Y_test_flat != 0
    y_pred_flat = y_pred_flat[non_zero_mask]
    Y_test_flat = Y_test_flat[non_zero_mask]
    
    diff = y_pred_flat - Y_test_flat
    std_dev = np.std(diff)
    return std_dev

def process_path(save_path, std_devs):
    for field in range(0, 151):
        field_str = f'{field:02d}'  # 格式化为两位数
        std_devs_for_field = []
        
        # 遍历同一组内的所有文件
        for i in range(10):  # 假设每组最多有10个文件
            predictions_file = f'predictions{field_str}.{i}.npy'
            truth_file = f'truth{field_str}.{i}.npy'
            
            # 检查文件是否存在
            if os.path.exists(os.path.join(save_path, predictions_file)) and os.path.exists(os.path.join(save_path, truth_file)):
                # 加载预测结果和真实值
                y_pred = np.load(os.path.join(save_path, predictions_file))
                Y_test = np.load(os.path.join(save_path, truth_file))
                
                # 计算标准差
                std_dev = calculate_std(y_pred, Y_test)
                std_devs_for_field.append(std_dev)
        
        # 计算平均标准差
        if std_devs_for_field:
            avg_std_dev = np.mean(std_devs_for_field)
            std_devs[field] = (avg_std_dev, np.std(std_devs_for_field))

# 处理四个路径的数据
process_path(save_path1, std_devs1)
process_path(save_path2, std_devs2)
process_path(save_path3, std_devs3)
process_path(save_path4, std_devs4)  # 新增路径处理

# 过滤掉None值
filtered_fields1 = np.array([field for field in range(0, 151) if field in std_devs1])
filtered_std_devs1 = np.array([std_devs1[field][0] for field in range(0, 151) if field in std_devs1])
filtered_errors1 = np.array([std_devs1[field][1] for field in range(0, 151) if field in std_devs1])

filtered_fields2 = np.array([field for field in range(0, 151) if field in std_devs2])
filtered_std_devs2 = np.array([std_devs2[field][0] for field in range(0, 151) if field in std_devs2])
filtered_errors2 = np.array([std_devs2[field][1] for field in range(0, 151) if field in std_devs2])

filtered_fields3 = np.array([field for field in range(0, 151) if field in std_devs3])
filtered_std_devs3 = np.array([std_devs3[field][0] for field in range(0, 151) if field in std_devs3])
filtered_errors3 = np.array([std_devs3[field][1] for field in range(0, 151) if field in std_devs3])

filtered_fields4 = np.array([field for field in range(0, 151) if field in std_devs4])  # 新增过滤
filtered_std_devs4 = np.array([std_devs4[field][0] for field in range(0, 151) if field in std_devs4])
filtered_errors4 = np.array([std_devs4[field][1] for field in range(0, 151) if field in std_devs4])

# 绘制标准差随磁场参数变化的图表
plt.figure(figsize=(6, 6))  
plt.errorbar(filtered_fields1, filtered_std_devs1, yerr=filtered_errors1, fmt='o', linestyle='-', color='b', label='cylindrical')
plt.errorbar(filtered_fields2, filtered_std_devs2, yerr=filtered_errors2, fmt='o', linestyle='-', color='r', label='cylindrical+')
plt.errorbar(filtered_fields3, filtered_std_devs3, yerr=filtered_errors3, fmt='o', linestyle='-', color='g', label='cylindrical++')
plt.errorbar(filtered_fields4, filtered_std_devs4, yerr=filtered_errors4, fmt='o', linestyle='-', color='m', label='cylindrical++B')  # 新增绘图
plt.xlabel('Magnetic Field Parameter')
plt.ylabel('Average Standard Deviation (std)')
plt.title('cylindrical VS cylindrical+ VS cylindrical++ VS cylindrical++B')
plt.grid(True)
plt.legend()

# 设置 y 轴范围从 0 开始
max_std_dev = max(max(filtered_std_devs1 + filtered_errors1), max(filtered_std_devs2 + filtered_errors2), max(filtered_std_devs3 + filtered_errors3), max(filtered_std_devs4 + filtered_errors4))
plt.ylim(0, max_std_dev * 1.1)

plt.savefig(os.path.join(save_path1, 'avg_std_vs_magnetic_field.png'), bbox_inches='tight')
plt.show()