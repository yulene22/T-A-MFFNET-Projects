#coding:utf-8
import math
import warnings
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter
warnings.filterwarnings("ignore")

#  假设采样频率为400hz,信号本身最大的频率为200hz，要滤除0.5hz以下，50hz以上频率成分，即截至频率为0.5hz，50hz
def butter_bandpass_filter(data, lowcut, highcut, samplingRate, order=5):
	nyq = 0.5 * samplingRate
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	y = lfilter(b, a, data)
	return y

# 微分熵计算
def compute_DE(data):
    variance = np.var(data, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


def decompose(filepath,name):

    # 读取数据
    data = loadmat(filepath)['EEG'][0][0][0]
    frequency = 200
    samples = data.shape[0]
    channels = data.shape[1]

    # 100个采样点计算一个微分熵
    num_sample = int(samples/100)

    bands = 5
    # 微分熵特征[14160, 17, 5]
    DE_Characteristics = np.empty([num_sample, channels, bands])

    temp_de = np.empty([0, num_sample])

    for channel in range(channels):

        trail_single = data[:, channel]

        Delta = butter_bandpass_filter(trail_single, 0.5, 4, frequency, order=3)
        Theta = butter_bandpass_filter(trail_single, 4, 8, frequency, order=3)
        Alpha = butter_bandpass_filter(trail_single, 8, 12, frequency, order=3)
        Beta = butter_bandpass_filter(trail_single, 12, 30, frequency, order=3)
        Gamma = butter_bandpass_filter(trail_single, 30, 50, frequency, order=3)

        DE_Delta = np.zeros(shape=[0], dtype=float)
        DE_Theta = np.zeros(shape=[0], dtype=float)
        DE_alpha = np.zeros(shape=[0], dtype=float)
        DE_beta = np.zeros(shape=[0], dtype=float)
        DE_gamma = np.zeros(shape=[0], dtype=float)

        # 依次计算5个频带的微分熵
        for index in range(num_sample):
            DE_Delta = np.append(DE_Delta, compute_DE(Delta[index * 100: (index + 1) * 100]))
            DE_Theta = np.append(DE_Theta, compute_DE(Theta[index * 100: (index + 1) * 100]))
            DE_alpha = np.append(DE_alpha, compute_DE(Alpha[index * 100: (index + 1) * 100]))
            DE_beta = np.append(DE_beta, compute_DE(Beta[index * 100: (index + 1) * 100]))
            DE_gamma = np.append(DE_gamma, compute_DE(Gamma[index * 100: (index + 1) * 100]))

        temp_de = np.vstack([temp_de, DE_Delta])
        temp_de = np.vstack([temp_de, DE_Theta])
        temp_de = np.vstack([temp_de, DE_alpha])
        temp_de = np.vstack([temp_de, DE_beta])
        temp_de = np.vstack([temp_de, DE_gamma])

    temp_trail_de = temp_de.reshape(-1, 5, num_sample)
    print("trail_DE shape", DE_Characteristics.shape)
    temp_trail_de = temp_trail_de.transpose([2, 0, 1])
    DE_Characteristics = np.vstack([temp_trail_de])

    print("trail_DE shape", DE_Characteristics.shape)
    return DE_Characteristics

filepath = 'D:/pytorch/Raw_Data/'
RawdataName = ['1_20151124_noon_2.mat', '2_20151106_noon.mat', '3_20151024_noon.mat', '4_20151105_noon.mat',
            '4_20151107_noon.mat', '5_20141108_noon.mat', '5_20151012_night.mat', '6_20151121_noon.mat',
            '7_20151015_night.mat', '8_20151022_noon.mat', '9_20151017_night.mat', '10_20151125_noon.mat',
            '11_20151024_night.mat', '12_20150928_noon.mat', '13_20150929_noon.mat', '14_20151014_night.mat',
            '15_20151126_night.mat', '16_20151128_night.mat', '17_20150925_noon.mat', '18_20150926_noon.mat',
            '19_20151114_noon.mat', '20_20151129_night.mat', '21_20151016_noon.mat']

EEGName = ['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08', 'EEG09', 'EEG010',
            'EEG011', 'EEG012', 'EEG013', 'EEG014', 'EEG015', 'EEG016', 'EEG017', 'EEG018', 'EEG019',
            'EEG020', 'EEG021', 'EEG022', 'EEG023']

DE = np.empty([0, 17, 5])

for i in range(len(RawdataName)):
    dataFile = filepath + RawdataName[i]
    print('processing {}'.format(RawdataName[i]))
    DE_Characteristics = decompose(dataFile, EEGName[i])
    DE = np.vstack([DE, DE_Characteristics])

# 保存数据
np.save("D:/pytorch/SEED-VIG/Raw_Data/dataset_DE/data_DE.npy", DE)