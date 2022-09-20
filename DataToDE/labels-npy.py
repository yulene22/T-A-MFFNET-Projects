import numpy as np
import scipy.io as sio

LF1 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/1_20151124_noon_2.mat'
LF2 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/2_20151106_noon.mat'
LF3 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/3_20151024_noon.mat'
LF4 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/4_20151105_noon.mat'
LF5 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/4_20151107_noon.mat'
LF6 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/5_20141108_noon.mat'
LF7 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/5_20151012_night.mat'
LF8 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/6_20151121_noon.mat'
LF9 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/7_20151015_night.mat'
LF10 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/8_20151022_noon.mat'
LF11 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/9_20151017_night.mat'
LF12 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/10_20151125_noon.mat'
LF13 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/11_20151024_night.mat'
LF14 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/12_20150928_noon.mat'
LF15 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/13_20150929_noon.mat'
LF16 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/14_20151014_night.mat'
LF17 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/15_20151126_night.mat'
LF18 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/16_20151128_night.mat'
LF19 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/17_20150925_noon.mat'
LF20 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/18_20150926_noon.mat'
LF21 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/19_20151114_noon.mat'
LF22 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/20_20151129_night.mat'
LF23 = 'C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/perclos_labels/21_20151016_noon.mat'

labels1 = list(sio.loadmat(LF1).values())[3]
labels2 = list(sio.loadmat(LF2).values())[3]
labels3 = list(sio.loadmat(LF3).values())[3]
labels4 = list(sio.loadmat(LF4).values())[3]
labels5 = list(sio.loadmat(LF5).values())[3]
labels6 = list(sio.loadmat(LF6).values())[3]
labels7 = list(sio.loadmat(LF7).values())[3]
labels8 = list(sio.loadmat(LF8).values())[3]
labels9 = list(sio.loadmat(LF9).values())[3]
labels10 = list(sio.loadmat(LF10).values())[3]
labels11 = list(sio.loadmat(LF11).values())[3]
labels12 = list(sio.loadmat(LF12).values())[3]
labels13 = list(sio.loadmat(LF13).values())[3]
labels14 = list(sio.loadmat(LF14).values())[3]
labels15 = list(sio.loadmat(LF15).values())[3]
labels16 = list(sio.loadmat(LF16).values())[3]
labels17 = list(sio.loadmat(LF17).values())[3]
labels18 = list(sio.loadmat(LF18).values())[3]
labels19 = list(sio.loadmat(LF19).values())[3]
labels20 = list(sio.loadmat(LF20).values())[3]
labels21 = list(sio.loadmat(LF21).values())[3]
labels22 = list(sio.loadmat(LF22).values())[3]
labels23 = list(sio.loadmat(LF23).values())[3]


L1 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
                labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
                labels20, labels21, labels22, labels23))




# 转换为.npy文件：
# 保存为numpy数组文件（.npy文件）
np.save("D:/pytorch/SEED-VIG/SEED-VIG/perclos_labels/labels-numpy_data.npy", L1)
# 读取numpy文件
# f = np.load("D:/pytorch/SEED-VIG/SEED-VIG/perclos_labels/labels-numpy_data.npy")
# print(f.shape)
# print(f)





