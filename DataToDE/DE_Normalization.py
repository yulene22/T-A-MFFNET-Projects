# coding:utf-8
# !usr/bin/env python
import numpy as np

def normalization(data):
    data = np.array(data)
    res_list = np.zeros((len(data[:, 0]), len(data[0])))
    for i in range(len(data[0])):
        max_min = np.max(data[:, i]) - np.min(data[:, i])  # ，每列最大值和最小值的差值
        min = np.min(data[:, i])
        for j in range(len(data[:, 0])):
            res_data = (data[j][i] - min) / max_min  # 每个值减去最小值 / 差值
            res_list[j][i] = res_data  # 存入数组
    return res_list


filepath = "D:/pytorch/SEED-VIG/Raw_Data/dataset_DE/data_DE.npy"
data = np.load(filepath)
data = data.reshape(1628400, 17)
DE_Normalization = normalization(data)
np.save("D:/pytorch/SEED-VIG/Raw_Data/dataset_DE/DE_Normalization.npy", DE_Normalization)