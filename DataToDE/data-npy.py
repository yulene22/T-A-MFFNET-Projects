import scipy.io as sio
import numpy as np

df1 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/1_20151124_noon_2.mat'
df2 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/2_20151106_noon.mat'
df3 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/3_20151024_noon.mat'
df4 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/4_20151105_noon.mat'
df5 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/4_20151107_noon.mat'
df6 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/5_20141108_noon.mat'
df7 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/5_20151012_night.mat'
df8 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/6_20151121_noon.mat'
df9 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/7_20151015_night.mat'
df10 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/8_20151022_noon.mat'
df11 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/9_20151017_night.mat'
df12 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/10_20151125_noon.mat'
df13 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/11_20151024_night.mat'
df14 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/12_20150928_noon.mat'
df15 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/13_20150929_noon.mat'
df16 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/14_20151014_night.mat'
df17 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/15_20151126_night.mat'
df18 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/16_20151128_night.mat'
df19 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/17_20150925_noon.mat'
df20 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/18_20150926_noon.mat'
df21 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/19_20151114_noon.mat'
df22 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/20_20151129_night.mat'
df23 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/21_20151016_noon.mat'



EEG1 = sio.loadmat(df1)['EEG'][0][0][0]
EEG2 = sio.loadmat(df2)['EEG'][0][0][0]
EEG3 = sio.loadmat(df3)['EEG'][0][0][0]
EEG4 = sio.loadmat(df4)['EEG'][0][0][0]
EEG5 = sio.loadmat(df5)['EEG'][0][0][0]
EEG6 = sio.loadmat(df6)['EEG'][0][0][0]
EEG7 = sio.loadmat(df7)['EEG'][0][0][0]
EEG8 = sio.loadmat(df8)['EEG'][0][0][0]
EEG9 = sio.loadmat(df9)['EEG'][0][0][0]
EEG10 = sio.loadmat(df10)['EEG'][0][0][0]
EEG11 = sio.loadmat(df11)['EEG'][0][0][0]
EEG12 = sio.loadmat(df12)['EEG'][0][0][0]
EEG13 = sio.loadmat(df13)['EEG'][0][0][0]
EEG14 = sio.loadmat(df14)['EEG'][0][0][0]
EEG15 = sio.loadmat(df15)['EEG'][0][0][0]
EEG16 = sio.loadmat(df16)['EEG'][0][0][0]
EEG17 = sio.loadmat(df17)['EEG'][0][0][0]
EEG18 = sio.loadmat(df18)['EEG'][0][0][0]
EEG19 = sio.loadmat(df19)['EEG'][0][0][0]
EEG20 = sio.loadmat(df20)['EEG'][0][0][0]
EEG21 = sio.loadmat(df21)['EEG'][0][0][0]
EEG22 = sio.loadmat(df22)['EEG'][0][0][0]
EEG23 = sio.loadmat(df23)['EEG'][0][0][0]


EEG = np.vstack((EEG1, EEG2, EEG3, EEG4, EEG5, EEG6, EEG7, EEG8, EEG9, EEG10,
                EEG11, EEG12, EEG13, EEG14, EEG15, EEG16, EEG17, EEG18, EEG19,
                EEG20, EEG21, EEG22, EEG23))


EEG_1 = EEG.reshape(20355, 17, 1600)
# print(EEG_1)

# 转换为.npy文件：
# 保存为numpy数组文件（.npy文件）
np.save("C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/Raw_Data-EEG_data.npy", EEG_1)
# # 读取numpy文件
# f = np.load("C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/Raw_Data-EEG_data.npy")
# print(f.shape)
