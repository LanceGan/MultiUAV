import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
DIS_THRESHOLD = 60 # m
SNR_THRESHOLD=3 #SIR threshold in Watt for outage/initial_1
np.random.seed(1)
# torch.manual_seed(1)
PB=0.1 #Transmit power in Watt
PN = -105   # dB
Fc=2e9 # Operating Frequency in GHz
LightSpeed=3*(10**8)
WaveLength=LightSpeed/(Fc) #wavelength in meter
step = 101  # include the start point at 0 and end point, the space between two sample points is D/(step-1)
D = 20

class Rural_world(object):
    def __init__(self,GT_loc):
        self.GT_loc = GT_loc
        self.HeightMapMatrix = self.Buliding_construct()
        
    def Buliding_construct(self):
        # ============== 生成山地区域地形（平滑高斯噪声） ==============
        np.random.seed(42)
        terrain_noise = np.random.randn(D * step, D * step) * 15  # 模拟地形起伏，尺度可调
        terrain_height = gaussian_filter(terrain_noise, sigma=10)  # 平滑以模拟连贯的山形
        terrain_height = np.clip(terrain_height, 0, 60)  # 限制山地高度范围在0-60m之间

        # ============== 生成建筑物（稀疏） ==============
        ALPHA = 0.2  # 建筑物占地系数
        BETA = 50    # 建筑物密度参数（适度减少以符合RMa）
        GAMA = 30    # 建筑物高度分布
        MAXHeight = 40
        MINHeight = 5

        N = int(BETA * (D ** 2))  # 建筑总数
        A = ALPHA * (D ** 2) / N  # 平均每栋建筑的占地面积 (km²)
        Side = np.sqrt(A)  # 每栋建筑占地边长 (km)

        # 建筑物高度采样
        np.random.seed(1)
        H_vec = np.random.rayleigh(GAMA, N)
        H_vec = np.clip(H_vec, MINHeight, MAXHeight)

        # 建筑物中心位置（均匀随机分布）
        XLOC = np.random.uniform(0, D, N)
        YLOC = np.random.uniform(0, D, N)

        # ============== 构建最终高度图 ==============
        HeightMapMatrix = terrain_height.copy()
        HeighMapArray = HeightMapMatrix.reshape(1, (D * step) ** 2) # (1,10201)
        for i in range(N):
            x1 = int(np.clip(np.floor((XLOC[i] - Side / 2) * (step - 1) / D), 0, step - 1))
            x2 = int(np.clip(np.ceil((XLOC[i] + Side / 2) * (step - 1) / D), 0, step - 1))
            y1 = int(np.clip(np.floor((YLOC[i] - Side / 2) * (step - 1) / D), 0, step - 1))
            y2 = int(np.clip(np.ceil((YLOC[i] + Side / 2) * (step - 1) / D), 0, step - 1))
            HeightMapMatrix[x1:x2 + 1, y1:y2 + 1] += H_vec[i]
        return HeightMapMatrix

    # ================
    # ================
    def getPointMiniOutage(self,loc_vec):
        PointLoc = loc_vec
        OutageMatrix,LoS_state,SNR_set = self.getPointOutageMatrix(PointLoc, SNR_THRESHOLD)
        return OutageMatrix,LoS_state,SNR_set


    def getPointOutageMatrix(self,PointLoc, SNR_th):
        numGT = len(self.GT_loc)
        SignalFromUav = []
        LoS_state = []
        TotalPower = 0
        for i in range(len(self.GT_loc)):
            GT = self.GT_loc[i, :]
            LoS = self.checkLoS(PointLoc, GT)
            LoS_state.append(LoS)
            MeasuredSignal = self.getReceivedPower_RicianAndRayleighFastFading(PointLoc, GT, LoS)
            SignalFromUav.append(MeasuredSignal)
            TotalPower = TotalPower + MeasuredSignal
        CoverMatrix = np.zeros(numGT)
        SNR_set = np.zeros(numGT)
        for i in range(len(self.GT_loc)):
            SignalTothisGT = SignalFromUav[i]
            SNR = SignalTothisGT / (10 ** (PN / 10))
            SNR_set[i] = SNR
            if SNR > SNR_th:
                CoverMatrix[i] = 1.0
        return CoverMatrix,LoS_state,SNR_set



    #This function check whether there is LoS between the GT and the given Loc

    def checkLoS(self,PointLoc,GT):
        SamplePoints=np.linspace(0,1,1000)
        XSample=GT[0]+SamplePoints*(PointLoc[0]-GT[0])
        YSample=GT[1]+SamplePoints*(PointLoc[1]-GT[1])
        ZSample=GT[2]+SamplePoints*(PointLoc[2]-GT[2])
        XRange = np.int_(np.floor(XSample * (step)))
        YRange = np.int_(np.floor(YSample * (step)))  #
        XRange = [max(x, 0) for x in XRange]  # remove the negative idex
        YRange = [max(x, 0) for x in YRange]  # remove the negative idex
        XRange = [min(x, 10000) for x in XRange]
        YRange = [min(x, 10000) for x in YRange]
        SelectedHeight = [self.HeightMapMatrix[XRange[i]][YRange[i]] for i in range(len(XRange))]
        if any([x > y for (x, y) in zip(SelectedHeight, ZSample)]):
            return False
        else:
            return True

    # 小尺度衰落
    # A simple fast-fading implementation: if LoS, Rician fading with K factor 15 dB; otherwise, Rayleigh fading
    def getReceivedPower_RicianAndRayleighFastFading(self,PointLoc, GT, LoS):
        LargeScale = self.getLargeScalePowerFromGT(PointLoc, GT,
                                              LoS)  # large-scale received power based on path loss
        # the random component, which is Rayleigh fading
        RayleighComponent = np.sqrt(0.5) * (
                    np.random.randn() + 1j * np.random.randn())

        if LoS:  # LoS, fast fading is given by Rician fading with K factor 15 dB
            K_R_dB = 15  # Rician K factor in dB
            K_R = 10 ** (K_R_dB / 10)
            AllFastFadingCoef = np.sqrt(K_R / (K_R + 1)) + np.sqrt(1 / (K_R + 1)) * RayleighComponent
        else:  # NLoS, fast fading is Rayleigh fading
            AllFastFadingCoef = RayleighComponent

        h_overall = AllFastFadingCoef * np.sqrt(LargeScale)
        PowerInstant = np.abs(h_overall) ** 2  # the instantneous received power in Watt
        return PowerInstant

    # 大尺度衰落
    def getLargeScalePowerFromGT(self,PointLoc,GT,LoS):  # pathloss
        ChGain=1.0
        Distance=100*np.sqrt((GT[0]-PointLoc[0])**2+(GT[1]-PointLoc[1])**2+(GT[2]-PointLoc[2])**2) #convert to meter
        if LoS:
            PathLoss_LoS_dB=0.1+20*np.log10(Distance)+20*np.log10(4*np.pi*Fc/LightSpeed)
            PathLoss_LoS_Linear=10**(-PathLoss_LoS_dB/10)
            Prx=ChGain*PB*PathLoss_LoS_Linear
        else:
            PathLoss_NLoS_dB=21+20*np.log10(Distance)+20*np.log10(4*np.pi*Fc/LightSpeed)
            PathLoss_NLoS_Linear=10**(-PathLoss_NLoS_dB/10)
            Prx=ChGain*PB*PathLoss_NLoS_Linear
        return Prx




