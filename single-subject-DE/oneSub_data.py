# coding:utf-8
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from Index_calculation import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Data = np.load('D:/pytorch/single-DE/singleDE-Normalization/dataDE-Nor23npy')
# print(Data.shape)
Data = Data.reshape(885, 17, 5, 16)
# print(Data.shape)
Data = torch.FloatTensor(Data)


# 标签
label = np.load('D:/pytorch/single-DE/single-label/EEGlabel23.npy')
# print(label.shape)
label = torch.FloatTensor(label)

# 用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签
# train_X, test_X为文本数据， train_Y, test_Y为标签数据
X_train, X_test, Y_train, Y_test = train_test_split(Data, label, test_size=0.3, random_state=800)

print("训练集测试集已划分完成............")
batch_size = 128
trainData = data.TensorDataset(X_train, Y_train)
testData = data.TensorDataset(X_test, Y_test)
train_dataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True,drop_last=True)
test_dataloader = DataLoader(testData, batch_size=batch_size, shuffle=True,drop_last=True)
print("dataloader已完成装载............")