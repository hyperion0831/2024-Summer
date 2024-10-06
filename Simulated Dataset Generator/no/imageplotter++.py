import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os

# 参数设置
PartMult = 1600  # 粒子数量
kNeutralPiFrac = 0.25  # 中性pi介子比例
kNeutralHadFrac = 0.5  # 中性强子比例
kMaxEta = 2.5
kBField_1=[0,5,10,25,35,50,75,90,100,120,140,150]
kNEvent = 10000  # 事件数量
kImSize = 56  # 图像尺寸
kResScale = 1.0
kTRKres = 0.05 * kResScale
kEMCres = 0.05 * kResScale
kHCALres = 0.1 * kResScale
kNonLin = 0.3

for kBField_2 in kBField_1:
    kBField=kBField_2/100
    for i in range(4):
        # 指定数据目录
        fDir = fr'/hy-tmp/{i}_B{kBField_2}_no'

        # 检查并创建目录（如果不存在）
        if not os.path.exists(fDir):
            os.makedirs(fDir)
        print(f"Write to {fDir}")

        # 初始化数组
        Charge = np.empty(PartMult)
        Hadron = np.empty(PartMult)
        Energy = np.empty(PartMult)
        eta = np.empty(PartMult)
        phi = np.empty(PartMult)
        WTruth = np.empty(PartMult)

        # 生成事件
        for n in range(kNEvent):
            if n % 100 == 0:
                print(f"Generating event {n}")
            
            for i in range(PartMult):
                Energy[i] = abs(np.random.normal() + np.random.normal(0, 2))
                WTruth[i] = Energy[i]
                x = np.random.rand()
                if x < kNeutralPiFrac:
                    Charge[i] = 0
                    Hadron[i] = 0
                elif x < kNeutralHadFrac:
                    Charge[i] = 0
                    Hadron[i] = 1
                else:
                    Charge[i] = 1 if np.random.rand() > 0.5 else -1
                    Hadron[i] = 1
                
                if Charge[i] == 0:
                    eta[i] = (np.random.rand() - 0.5) * 2 * kMaxEta
                    phi[i] = (np.random.rand() - 0.5) * 2 * np.pi
                else:
                    eta[i] = (np.random.rand() - 0.5) * 2.0 * kMaxEta  # 修改为2.0
                    phi[i] = (np.random.rand() - 0.5) * 2.0 * np.pi  # 修改为2.0
            
            # 生成真值图像
            c_truth, xe, ye = np.histogram2d(eta, phi, weights=WTruth,
                                             range=[[-kMaxEta, kMaxEta], [-np.pi, np.pi]],
                                             bins=(kImSize, kImSize))
            
            # 生成跟踪器图像
            kMinTrackE = 0.1
            WTrkP = np.zeros(PartMult)
            WTrkN = np.zeros(PartMult)
            for i in range(PartMult):
                if Charge[i] > 0 and Energy[i] > kMinTrackE:
                    WTrkP[i] = Energy[i] * np.random.normal(1, kTRKres)
                elif Charge[i] < 0 and Energy[i] > kMinTrackE:
                    WTrkN[i] = Energy[i] * np.random.normal(1, kTRKres)

            c_trkp, xe, ye = np.histogram2d(eta, phi, weights=WTrkP,
                                            range=[[-kMaxEta, kMaxEta], [-np.pi, np.pi]],
                                            bins=(kImSize, kImSize))
            
            c_trkn, xe, ye = np.histogram2d(eta, phi, weights=WTrkN,
                                            range=[[-kMaxEta, kMaxEta], [-np.pi, np.pi]],
                                            bins=(kImSize, kImSize))
            
            # 磁场影响下的phi角度修正
            for i in range(PartMult):
                if Charge[i] > 0:
                    phi[i] = phi[i] + kBField * 1 / (Energy[i])
                elif Charge[i] < 0:
                    phi[i] = phi[i] - kBField * 1 / (Energy[i])
            
            
            # 生成电磁量能器图像
            WEmcal = np.zeros(PartMult)
            kMinEmcalE = 0.2
            for i in range(PartMult):
                if Hadron[i] == 0 and Energy[i] > kMinEmcalE:
                    WEmcal[i] = (Energy[i] - kNonLin * np.sqrt(Energy[i])) * 0.9 * np.random.normal(1, kEMCres)
                    Energy[i] = Energy[i] * 0.1
                elif Energy[i] > kMinEmcalE:
                    WEmcal[i] = Energy[i] * 0.1 * np.random.normal(1, kEMCres)
                    Energy[i] = Energy[i] * 0.9

            # 生成强子量能器图像
            WHcal = np.zeros(PartMult)
            kMinHcalE = 0.3
            for i in range(PartMult):
                if Energy[i] > kMinHcalE:
                    WHcal[i] = (Energy[i] - kNonLin * np.sqrt(Energy[i])) * 0.9 * np.random.normal(1, kHCALres)
                    Energy[i] = Energy[i] * 0.1
            
            c_emcal, xe, ye = np.histogram2d(eta, phi, weights=WEmcal,
                                             range=[[-kMaxEta, kMaxEta], [-np.pi, np.pi]],
                                             bins=(kImSize, kImSize))

            c_hcal, xe, ye = np.histogram2d(eta, phi, weights=WHcal,
                                            range=[[-kMaxEta, kMaxEta], [-np.pi, np.pi]],
                                            bins=(kImSize, kImSize))

            # 保存图像文件
            io.imsave(os.path.join(fDir, f"truth_{n}.tiff"), c_truth.astype(np.float32))
            io.imsave(os.path.join(fDir, f"trkp_{n}.tiff"), c_trkp.astype(np.float32))
            io.imsave(os.path.join(fDir, f"trkn_{n}.tiff"), c_trkn.astype(np.float32))
            io.imsave(os.path.join(fDir, f"emcal_{n}.tiff"), c_emcal.astype(np.float32))
            io.imsave(os.path.join(fDir, f"hcal_{n}.tiff"), c_hcal.astype(np.float32))

            # 显示第一个事件的图像
            if n == 0:
                plt.imshow(c_truth)
                plt.title("truth")
                plt.show()
                plt.imshow(c_trkp)
                plt.title("trkp")
                plt.show()
                plt.imshow(c_trkn)
                plt.title("trkn")
                plt.show()
                plt.imshow(c_emcal)
                plt.title("emcal")
                plt.show()
                plt.imshow(c_hcal)
                plt.title("hcal")
                plt.show()

        print("数据生成完成")