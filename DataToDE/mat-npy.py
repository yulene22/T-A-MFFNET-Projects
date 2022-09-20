# coding:utf-8
import numpy as np
import scipy.io as sio

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
EEG_1 = EEG1.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG1.npy", EEG_1)
EEG2 = sio.loadmat(df2)['EEG'][0][0][0]
EEG_2 = EEG2.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG2.npy", EEG_2)
EEG3 = sio.loadmat(df3)['EEG'][0][0][0]
EEG_3 = EEG3.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG3.npy", EEG_3)
EEG4 = sio.loadmat(df4)['EEG'][0][0][0]
EEG_4 = EEG4.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG4.npy", EEG_4)
EEG5 = sio.loadmat(df5)['EEG'][0][0][0]
EEG_5 = EEG5.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG5.npy", EEG_5)
EEG6 = sio.loadmat(df6)['EEG'][0][0][0]
EEG_6 = EEG6.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG6.npy", EEG_6)
EEG7 = sio.loadmat(df7)['EEG'][0][0][0]
EEG_7 = EEG7.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG7.npy", EEG_7)
EEG8 = sio.loadmat(df8)['EEG'][0][0][0]
EEG_8 = EEG8.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG8.npy", EEG_8)
EEG9 = sio.loadmat(df9)['EEG'][0][0][0]
EEG_9 = EEG9.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG9.npy", EEG_9)
EEG10 = sio.loadmat(df10)['EEG'][0][0][0]
EEG_10 = EEG10.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG10.npy", EEG_10)
EEG11 = sio.loadmat(df11)['EEG'][0][0][0]
EEG_11 = EEG11.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG11.npy", EEG_11)
EEG12 = sio.loadmat(df12)['EEG'][0][0][0]
EEG_12 = EEG12.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG12.npy", EEG_12)
EEG13 = sio.loadmat(df13)['EEG'][0][0][0]
EEG_13 = EEG13.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG13.npy", EEG_13)
EEG14 = sio.loadmat(df14)['EEG'][0][0][0]
EEG_14 = EEG14.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG14.npy", EEG_14)
EEG15 = sio.loadmat(df15)['EEG'][0][0][0]
EEG_15 = EEG15.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG15.npy", EEG_15)
EEG16 = sio.loadmat(df16)['EEG'][0][0][0]
EEG_16 = EEG16.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG16.npy", EEG_16)
EEG17 = sio.loadmat(df17)['EEG'][0][0][0]
EEG_17 = EEG17.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG17.npy", EEG_17)
EEG18 = sio.loadmat(df18)['EEG'][0][0][0]
EEG_18 = EEG18.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG18.npy", EEG_18)
EEG19 = sio.loadmat(df19)['EEG'][0][0][0]
EEG_19 = EEG19.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG19.npy", EEG_19)
EEG20 = sio.loadmat(df20)['EEG'][0][0][0]
EEG_20 = EEG20.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG20.npy", EEG_20)
EEG21 = sio.loadmat(df21)['EEG'][0][0][0]
EEG_21 = EEG21.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG21.npy", EEG_21)
EEG22 = sio.loadmat(df22)['EEG'][0][0][0]
EEG_22 = EEG22.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG22.npy", EEG_22)
EEG23 = sio.loadmat(df23)['EEG'][0][0][0]
EEG_23 = EEG23.reshape(17, 1416000)
np.save("D:/pytorch/SEED-VIG/Raw_Data/EEG_npy/EEG23.npy", EEG_23)