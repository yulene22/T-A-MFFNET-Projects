# coding:utf-8
import torch
from sklearn.model_selection import train_test_split
import torch.utils.data as data
from torch.utils.data import DataLoader
from Index_calculation import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G = testclass()
# 数据集
# 原始信号
# rawData = np.load("C:/Users/admin/PycharmProjects/pythonProject/pytorch1/SEED-VIG/Raw_Data/Raw_Data-EEG_data.npy")
# print(rawData.shape)
# rawData = rawData.reshape(20355,17,10,160)
# rawData = torch.FloatTensor(rawData)
# rawData = rawData.unsqueeze(1)

# 归一化后的数据集
# normalization_rawdata = np.load("D:/pytorch/SEED-VIG/Raw_Data/Raw_Data_Normalization/Raw_Data_Normalization.npy")
# normalization_rawdata = normalization_rawdata.reshape(20355,17,10,160)
# normalization_rawdata = torch.FloatTensor(normalization_rawdata)
# normalization_rawdata = normalization_rawdata.unsqueeze(1)

# 带通滤波之后的数据集
# filter_data = np.load('D:/pytorch/SEED-VIG/Raw_Data/filter_data/filter_EEG_data.npy')
# filter_data = filter_data.reshape(20355,17,10,160)
# filter_data = torch.FloatTensor(filter_data)
# filter_data = filter_data.unsqueeze(1)

# 带通滤波 + 归一化之后的数据集
# filter_normalization_data = np.load('D:/pytorch/SEED-VIG/Raw_Data/filter_normalizayion/filter_normalizayion.npy')
# filter_normalization_data = filter_normalization_data.reshape(20355,17,10,160)
# filter_normalization_data = torch.FloatTensor(filter_normalization_data)
# filter_normalization_data = filter_normalization_data.unsqueeze(1)

# 提取微分熵特征之后的数据集
# DE_data = np.load("D:/pytorch/SEED-VIG/Raw_Data/dataset_DE/data_DE.npy")
# print(DE_data.shape)

# DE_data = DE_data.reshape(20355, 17, 80)
# DE_data = DE_data.reshape(20355, 17, 5, 16)
# print(DE_data.shape)
# DE_data = torch.FloatTensor(DE_data)
# # DE_data = DE_data.unsqueeze(1)

# 微分熵 + 归一化
DE_Normalization = np.load("D:/pytorch/SEED-VIG/Raw_Data/dataset_DE/DE_Normalization.npy")
# DE_Normalization = np.load("D:/pytorch/single-DE/singleDE-Normalization/dataDE-Nor22.npy")
# DE_Normalization = DE_Normalization.reshape(885, 17, 5, 16)
# DE_Normalization = DE_Normalization.reshape(20355, 17, 5, 16)
DE_Normalization = DE_Normalization.reshape(20355, 17, 80)
DE_Normalization = torch.FloatTensor(DE_Normalization)
# DE_Normalization = DE_Normalization.unsqueeze(1)


# 标签
label = np.load('D:/pytorch/SEED-VIG/SEED-VIG/perclos_labels/labels-numpy_data.npy')
# label = np.load('D:/pytorch/single-DE/single-label/EEGlabel22.npy')
# print(label.shape)
label = torch.FloatTensor(label)

# 用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签
# train_X, test_X为文本数据， train_Y, test_Y为标签数据
X_train, X_test, Y_train, Y_test = train_test_split(DE_Normalization, label, test_size=0.3, random_state=1600)

print("训练集测试集已划分完成............")
batch_size = 128
trainData = data.TensorDataset(X_train, Y_train)
testData = data.TensorDataset(X_test, Y_test)
train_dataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True,drop_last=True)
test_dataloader = DataLoader(testData, batch_size=batch_size, shuffle=True,drop_last=True)
print("dataloader已完成装载............")