import scipy.io as sio
import numpy as np


filepath = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/'
RawdataName = ['1_20151124_noon_2.mat', '2_20151106_noon.mat', '3_20151024_noon.mat', '4_20151105_noon.mat',
            '4_20151107_noon.mat', '5_20141108_noon.mat', '5_20151012_night.mat', '6_20151121_noon.mat',
            '7_20151015_night.mat', '8_20151022_noon.mat', '9_20151017_night.mat', '10_20151125_noon.mat',
            '11_20151024_night.mat', '12_20150928_noon.mat', '13_20150929_noon.mat', '14_20151014_night.mat',
            '15_20151126_night.mat', '16_20151128_night.mat', '17_20150925_noon.mat', '18_20150926_noon.mat',
            '19_20151114_noon.mat', '20_20151129_night.mat', '21_20151016_noon.mat']

EEGName = ['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08', 'EEG09', 'EEG010',
            'EEG011', 'EEG012', 'EEG013', 'EEG014', 'EEG015', 'EEG016', 'EEG017', 'EEG018', 'EEG019',
            'EEG020', 'EEG021', 'EEG022', 'EEG023']


for i in range(len(RawdataName)):
    dataFile = filepath + RawdataName[i]
    print('processing {}'.format(RawdataName[i]))
    EEGName[i] = sio.loadmat(dataFile)['EEG'][0][0][0]
    EEG = EEGName[i]
    np.save("D:/pytorch/single-DE/single-EEG/data"+str(i)+".npy", EEG)