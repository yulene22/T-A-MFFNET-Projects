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

filepath = 'D:/pytorch/single-DE/single-DE/'
RawdataName = ['dataDE0.npy', 'dataDE1.npy', 'dataDE2.npy', 'dataDE3.npy',
            'dataDE4.npy', 'dataDE5.npy', 'dataDE6.npy', 'dataDE7.npy',
            'dataDE8.npy', 'dataDE9.npy', 'dataDE10.npy', 'dataDE11.npy',
            'dataDE12.npy', 'dataDE13.npy', 'dataDE14.npy', 'dataDE15.npy',
            'dataDE16.npy', 'dataDE17.npy', 'dataDE18.npy', 'dataDE19.npy',
            'dataDE20.npy', 'dataDE21.npy', 'dataDE22.npy']


for i in range(len(RawdataName)):
    dataFile = filepath + RawdataName[i]
    data = np.load(dataFile)
    data = data.reshape(70800, 17)
    print('processing {}'.format(RawdataName[i]))
    DE_Normalization = normalization(data)
    np.save("D:/pytorch/single-DE/singleDE-Normalization/dataDE-Nor"+str(i)+".npy", DE_Normalization)

