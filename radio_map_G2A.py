import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter
print("Here we go G2A!")
# ================================
# 3GPP Rural Macro (RMa) Building Distribution Model
# ================================

D = 4  # 区域边长 D x D km²
step = 101  # 分辨率为 D / (step-1) km

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
    

# BS_loc=np.array([[0.477,0.456,0.025],[1.489,0.489,0.025],[0.463,1.378,0.025],[1.348,1.489,0.025]]) # 2kmx2km area, 3 BSs
BS_loc=np.array([[1.503,0.827,0.025],[2.698,0.825,0.025],[0.743,2.036,0.025],[2.001,2.036,0.025],[3.247,2.036,0.025],[1.510,3.248,0.025],[2.702,3.248,0.025]]) # 4kmx4km area, 7 BSs

BS_thetaD = 120  # The downtile angle in degree [0, 180]
PB = 0.1 # BS传输功率
Fc = 2  # Operating Frequency in GHz
SIR_THRESHOLD=3 #SIR threshold in dB for outage


LightSpeed = 3 * (10 ** 8)
WaveLength = LightSpeed / (Fc * (10 ** 9))  # wavelength in meter 波长
SectorBoreSiteAngle = [-120, 0, 120]  # the sector angle for each BS每个 BS 的扇区角
Sec_num = np.size(SectorBoreSiteAngle)  # 每个小区的扇区数
FastFadingSampleSize = 1000  # number of signal measurements per time step每个时间步的信号测量次数


# 存储天线增益
def getAntennaGain(Theta_deg, Phi_deg):
    # Basic Setting about Antenna Arrays
    ArrayElement_Horizontal = 1  # number of array elements in horizontal水平方向的数组元素数
    ArrayElement_Vertical = 8
    DV = 0.5 * WaveLength  # spacing for vertical array垂直阵列的间距
    DH = 0.5 * WaveLength  # spacing for horizontal array水平阵列间距
    angleTiltV = BS_thetaD
    angleTiltH = 0  # horizontal tilt angle水平倾斜角
    # Find the element power gain求元素功率增益
    angle3dB = 65
    Am = 30
    AH = -np.min([12 * (Phi_deg / angle3dB) ** 2, Am])  # element power gain in horizontal水平元件功率增益
    AV = -np.min([12 * ((Theta_deg - 90) / angle3dB) ** 2, Am])  # 垂直元件功率增益
    Gemax = 8  # dBi antenna gain in dB above an isotropic radiator, Maximum directional gain of an antenna elementdBi 天线增益，以 dB 为单位，高于各向同性辐射器，天线元件的最大方向增益
    Aelement = -np.min([-(AH + AV), Am])
    GelementdB = Gemax + Aelement  # dBi
    Gelement = 10 ** (GelementdB / 10)
    Felement = np.sqrt(Gelement)
    # Find array gain
    k = 2 * np.pi / WaveLength  # wave number
    kVector = k * np.array([np.sin(Theta_deg / 180 * np.pi) * np.cos(Phi_deg / 180 * np.pi),
                            np.sin(Theta_deg / 180 * np.pi) * np.sin(Phi_deg / 180 * np.pi),
                            np.cos(Theta_deg / 180 * np.pi)])  # wave vector
    rMatrix = np.zeros(shape=(ArrayElement_Horizontal * ArrayElement_Vertical, 3))
    for n in range(ArrayElement_Horizontal):
        rMatrix[(n + 1) * np.arange(ArrayElement_Vertical), 2] = np.arange(ArrayElement_Vertical) * DV
        rMatrix[(n + 1) * np.arange(ArrayElement_Vertical), 1] = n * DH
    SteeringVector = np.exp(-1j * (rMatrix.dot(np.transpose(kVector))))
    # Vertical Weight Vector
    Weight_Vertical = (1 / np.sqrt(ArrayElement_Vertical)) * np.exp(
        -1j * k * np.arange(ArrayElement_Vertical) * DV * np.cos(angleTiltV / 180 * np.pi))
    Weight_Horizontal = (1 / np.sqrt(ArrayElement_Horizontal)) * np.exp(
        -1j * k * np.arange(ArrayElement_Horizontal) * DH * np.sin(angleTiltH / 180 * np.pi))
    Weight2D = np.kron(Weight_Horizontal, np.transpose(Weight_Vertical))
    WeightFlatten = Weight2D.reshape(1, ArrayElement_Vertical * ArrayElement_Horizontal)
    ArrayFactor = np.conjugate(WeightFlatten).dot(SteeringVector.reshape(ArrayElement_Vertical, 1))
    Farray = Felement * ArrayFactor
    Garray = (np.abs(Farray)) ** 2
    return 10 * np.log10(Garray), Farray

#=========Main Function that determines the best outage from all BS at a given location=======
#loc_vec: a matrix, nx3, each row is a (x,y,z) location
#SIR_th: the SIR threshold for determining outage
def getPointMiniOutage(loc_vec):
    numLoc=len(loc_vec)
    Out_vec=[]
    Out_vec_SINR=[]
    for i in range(numLoc):
        PointLoc=loc_vec[i,:]
        OutageMatrix,OutageMatrixSINR=getPointOutageMatrix(PointLoc,SIR_THRESHOLD)
        MiniOutage=np.min(OutageMatrix)
        MiniOutageSINR = np.max(OutageMatrixSINR)
        Out_vec.append(MiniOutage)
        Out_vec_SINR.append(MiniOutageSINR)
    return Out_vec,Out_vec_SINR

#For a given location, return the empirical outage probaibility from all sectors of all BSs
#PointLoc:  the given point location
#SIR_th: the SIR threshold for defining the outage 
#OutageMatrix: The average outage probability for connecting with each site, obtained by averaging over all the samples

def getPointOutageMatrix(PointLoc,SIR_th):
    numBS=len(BS_loc)
    SignalFromBS=[]
    TotalPower=0

    for i in range(len(BS_loc)):
        BS=BS_loc[i,:]
        LoS=checkLoS(PointLoc,BS)
        MeasuredSignal=getReceivedPower_RicianAndRayleighFastFading(PointLoc,BS,LoS)
        SignalFromBS.append(MeasuredSignal)
        TotalPower=TotalPower+MeasuredSignal
    TotalPowerAllSector=np.sum(TotalPower, axis=1) #the interference of all power
    OutageMatrix=np.zeros(shape=(numBS,Sec_num))
    SIR_dB_avg = np.zeros(shape=(numBS,Sec_num))
    
    for i in range(len(BS_loc)):
        SignalFromThisBS=SignalFromBS[i]
        for sector in range(Sec_num):
            SignalFromThisSector=SignalFromThisBS[:,sector]
            #--------------------主要改这一项-----------------------#
            SIR=SignalFromThisSector/(TotalPowerAllSector-SignalFromThisSector+10 ** ((-140 - 30) / 10)) # dB
            SIR_dB=10*np.log10(SIR)
            SIR_dB_avg[i, sector] = np.sum(SIR_dB) / len(SIR_dB)
            # print(np.sum(SIR_dB<SIR_th))
            # print(len(SIR_dB))
            OutageMatrix[i,sector]=np.sum(SIR_dB<SIR_th)/len(SIR_dB)
            # print(OutageMatrix[i,sector])
    return OutageMatrix,SIR_dB_avg



def getLargeScalePowerFromBS(PointLoc, BS, Theta_deg, Phi_deg, LoS, Fc=2.0, H_u=100, SectorBoreSiteAngle=[0, 120, -120]):
    """
    Calculate large-scale received power (linear scale) from BS using RMa model (3GPP TR 36.814)

    Parameters:
        PointLoc: [x_uav, y_uav, h_uav] in km
        BS: [x_bs, y_bs, h_bs] in km
        Theta_deg: elevation angle (for antenna)
        Phi_deg: azimuth angle (for antenna)
        LoS: boolean indicating if LoS exists
        Fc: carrier frequency in GHz
        H_u: UAV height in meters
        SectorBoreSiteAngle: list of sector directions

    Returns:
        Prx: received power (linear scale)
    """
    Sector_num = len(SectorBoreSiteAngle)
    Phi_Sector_ref = Phi_deg - np.array(SectorBoreSiteAngle)
    Phi_Sector_ref[Phi_Sector_ref < -180] += 360
    Phi_Sector_ref[Phi_Sector_ref > 180] -= 360

    ChGain_dB = np.zeros((1, Sector_num))
    for i in range(Sector_num):
        ChGain_dB[0, i], _ = getAntennaGain(Theta_deg, Phi_Sector_ref[i])
    ChGain=np.power(10,ChGain_dB/10)  # convert from dB to linear

    # Compute 3D distance in meters
    dx = (BS[0] - PointLoc[0]) 
    dy = (BS[1] - PointLoc[1]) 
    dz = (BS[2] - PointLoc[2]) 
    d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)*1000  # in meters

    H_m = BS[2] * 1000  # BS height in meters

    # PL^{LoS} according to your formula
    PL_LoS = 20 * np.log10(40 * np.pi * d * Fc / 3) + \
             min(0.03 * H_u ** 1.72, 10) * np.log10(d) - \
             0.58 * np.log10(H_m)
             
    PL_NLoS = 161.04 - 7.1 * np.log10(20) + 7.5 * np.log10(H_m) - \
                       (24.37 - 3.7 * (H_m / H_u) ** 2) * np.log10(H_m) + \
                       (43.42 - 3.1 * np.log10(H_m)) * (np.log10(d) - 3) + \
                       20 * np.log10(Fc) - (3.2 * (np.log10(11.75 * H_m)) ** 2 - 4.97)
    if LoS:
        PathLoss_dB = PL_LoS
    else:      
        PathLoss_dB = PL_NLoS

    # Convert path loss to linear scale
    PathLoss_linear = 10 ** (-PathLoss_dB / 10)

    # Compute received power (you need to define PB externally)
    Prx = ChGain * PB * PathLoss_linear  # [1, Sector_num]

    return Prx
 
# Return the received power at a location from all the three sectors of a BS
# While the large scale path loss power is a constant for given location and site, the fast fading may change very fast.
# Hence, we return multiple fast fading coefficients. The number of samples is determined by FastFadingSampleSize
# A simple fast-fading implementation: if LoS, Rician fading with K factor 15 dB; otherwise, Rayleigh fading
def getReceivedPower_RicianAndRayleighFastFading(PointLoc, BS, LoS):
    HorizonDistance = np.sqrt((BS[0] - PointLoc[0]) ** 2 + (BS[1] - PointLoc[1]) ** 2)
    Theta = np.arctan((BS[2] - PointLoc[2]) / HorizonDistance)  # elevation angle
    Theta_deg = np.rad2deg(Theta) + 90  # convert to the (0,180) degree
    if (PointLoc[1] == BS[1]) & (PointLoc[0] == BS[0]):
        Phi = 0
    else:
        Phi = np.arctan((PointLoc[1] - BS[1]) / (PointLoc[0] - BS[0] + 0.00001))  # to avoid dividing by 0
    Phi_deg = np.rad2deg(Phi)
    # Convert the horizontal degree to the range (-180,180)
    if (PointLoc[1] > BS[1]) & (PointLoc[0] < BS[0]):
        Phi_deg = Phi_deg + 180
    elif (PointLoc[1] < BS[1]) & (PointLoc[0] < BS[0]):
        Phi_deg = Phi_deg - 180
    LargeScale = getLargeScalePowerFromBS(PointLoc, BS, Theta_deg, Phi_deg,
                                          LoS)  # large-scale received power based on path loss

    # the random component, which is Rayleigh fading
    RayleighComponent = np.sqrt(0.5) * (
                np.random.randn(FastFadingSampleSize, 3) + 1j * np.random.randn(FastFadingSampleSize, 3))  # (1000,3)
    # print("瑞丽数组大小",RayleighComponent.shape)

    if LoS:  # LoS, fast fading is given by Rician fading with K factor 15 dB
        K_R_dB = 15  # Rician K factor in dB
        K_R = 10 ** (K_R_dB / 10)
        threeD_distance = 1000 * np.sqrt((BS[0] - PointLoc[0]) ** 2 + (BS[1] - PointLoc[1]) ** 2 + (
                    BS[2] - PointLoc[2]) ** 2)  # 3D distance in meter
        DetermComponent = np.exp(-1j * 2 * np.pi * threeD_distance / WaveLength)  # deterministic component
        AllFastFadingCoef = np.sqrt(K_R / (K_R + 1)) * DetermComponent + np.sqrt(1 / (K_R + 1)) * RayleighComponent
    else:  # NLoS, fast fading is Rayleigh fading
        AllFastFadingCoef = RayleighComponent

    h_overall = AllFastFadingCoef * np.sqrt(np.tile(LargeScale, (FastFadingSampleSize, 1)))
    PowerInstant = np.abs(h_overall) ** 2  # the instantneous received power in Watt
    # print("瞬时功率",PowerInstant.shape)
    return PowerInstant

# This function check whether there is LoS between the BS and the given Loc三维坐标
# 概率LoS模型
# def checkLoS(PointLoc, BS):

#     # Calculate 3D distance
#     dx = PointLoc[0] - BS[0]
#     dy = PointLoc[1] - BS[1]
#     dz = PointLoc[2] - BS[2]
#     d = np.sqrt(dx**2 + dy**2 + dz**2)*1000

#     # LoS probability
#     if d < 10:
#         p_los = 1.0
#     else:
#         p_los = np.exp(-d/ 1000.0)

    
#     p = [p_los, 1 - p_los]
#     # Randomly choose LoS or NLoS based on the probability
#     LoS = np.random.choice([True, False], p=p)  
#     print(f"LoS probability: {p_los}, LoS: {LoS}")
#     return LoS

# 原论文中的 LoS 检查函数
def checkLoS(PointLoc, BS):
    SamplePoints = np.linspace(0, 1, 100) # (0,1)100个点
    XSample = BS[0] + SamplePoints * (PointLoc[0] - BS[0])
    YSample = BS[1] + SamplePoints * (PointLoc[1] - BS[1])
    ZSample = BS[2] + SamplePoints * (PointLoc[2] - BS[2])
    XRange = np.floor(XSample * (step - 1)) # 地板除
    YRange = np.floor(YSample * (step - 1))
    XRange = [max(x, 0) for x in XRange]  # remove the negative idex去掉负号
    YRange = [max(x, 0) for x in YRange]  # remove the negative idex
    Idx_vec = np.int_((np.array(XRange) * D * step + np.array(YRange))) # 高度数组里的下标
    SelectedHeight = [HeighMapArray[0, i] for i in Idx_vec]  # 比较高度
    if any([x > y for (x, y) in zip(SelectedHeight, ZSample)]): # 建筑物高度大于样点高度
        # print("NLoS")
        return False
    else:
        # print("LoS")
        return True

def getPointDateRate(loc_vec):  # 输入：n个坐标点的坐标，nx3矩阵，返回：每个点的中断率1x201数组
    numLoc = len(loc_vec)  # numLoc = n
    Out_vec = []
    Out_SINR_vec = []
    
    for i in range(numLoc):  # 循环n次
        PointLoc = loc_vec[i, :]
        OutageMatrix,SIR_dB_avg = getPointOutageMatrix(PointLoc, SIR_THRESHOLD)
        MiniOutage = np.min(OutageMatrix) #取出中断概率最小的基站的信号
        MaxSINR = np.max(SIR_dB_avg)
        Out_SINR_vec.append(MaxSINR)  # 1x201数组
        # 吞吐量
    return MaxSINR #返回的是信噪比最大的值   

##============VIew the radio map for given height
if __name__ == '__main__':
   
    step = 101
    UAV_height=0.100 # UAV height in km
    X_vec=range(D*(step-1)+1)
    Y_vec=range(D*(step-1)+1)
    numX,numY=np.size(X_vec),np.size(Y_vec)

    OutageMapActual=np.zeros(shape=(numX,numY))
    OutageMapActualSINR=np.zeros(shape=(numX,numY))
    Loc_vec_All=np.zeros(shape=(numX*numY,3))

    for i in range(numX):
        Loc_vec=np.zeros(shape=(numY,3))
        Loc_vec[:,0]=X_vec[i]/step
        Loc_vec[:,1]=np.array(Y_vec)/step
        Loc_vec[:,2]=UAV_height
        Loc_vec_All[i*numY:(i+1)*numY,:]=Loc_vec
        
        OutageMapActual[:,i],OutageMapActualSINR[:,i] = getPointMiniOutage(Loc_vec)
        
    Outage_vec_All = np.reshape(OutageMapActual, numX * numY)

   
    fig = plt.figure(10)
    plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, OutageMapActualSINR)
    v = np.arange(-10, 24, 2)
    cbar = plt.colorbar(ticks=v)
    cbar.set_label('SINR', labelpad=20, rotation=270, fontsize=14)
    plt.xlabel('x (meter)', fontsize=14)
    plt.ylabel('y (meter)', fontsize=14)
    plt.savefig('results/figs/OutageMapActualSINR.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    fig = plt.figure(11)
    plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, 1-OutageMapActual)
    v = np.linspace(0, 1, 11, endpoint=True)    
    cbar = plt.colorbar(ticks=v)
    cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)
    plt.xlabel('x (meter)', fontsize=14)
    plt.ylabel('y (meter)', fontsize=14)
    plt.savefig('results/figs/OutageMapActual.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    np.savez('results/datas/Radio_datas', OutageMapActual, OutageMapActualSINR ,X_vec, Y_vec)