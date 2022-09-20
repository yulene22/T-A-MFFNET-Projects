#coding=utf-8
import numpy as np

class testclass:
    # 二分类
    def train_lable2(self, x):
        a = []
        for i in range(len(x)):
            if x[i] <= 0.35:
                a.append(0)
            else:
                a.append(1)
        return a

    # 三分类
    # def train_lable3(x):
    #     a = []
    #     for i in range(len(x)):
    #         if x[i] < 0.35:
    #             a.append(0)
    #         elif x[i] < 0.7:
    #             a.append(1)
    #         else:
    #             a.append(2)
    #     return a

    def acc(self,x_lable, y_lable):
        b = []
        for i in range(len(x_lable)):
            if x_lable[i] == y_lable[i]:
                b.append(1)
        train_acc = len(b)
        return train_acc

    def len(self,X, batch_size):
        if(X%batch_size == 0):
            len = X
        else:
            len = (X - (X % batch_size))

        return len

    def nums(self,data):
        nums = 0
        for i in range(len(data)):
            for j in range(len(data[0])):
                if(data[i][j] <= 0.35):
                    nums = nums+1
        return nums

    # TP=matrix[0] # 实际为正，预测为正
    # TN=matrix[1] # 实际为负，预测为负
    # FP=matrix[2] # 实际为负，预测为正
    # FN=matrix[3] # 实际为正，预测为负
    # def Compute_TP(self, test_label, label, tp):
    #     TP = tp
    #     for i in range(len(test_label)):
    #         if test_label[i] == 0 and label[i] == 0:
    #             TP = TP + 1
    #     return TP
    #
    # def Compute_TN(self, test_label, label, tn):
    #     TN = tn
    #     for i in range(len(test_label)):
    #         if(test_label[i] == 1 and label[i] == 1):
    #             TN += 1
    #     return TN
    #
    # def Compute_FP(self, test_label, label, fp):
    #     FP = fp
    #     for i in range(len(test_label)):
    #         if(test_label[i] == 1 and label[i] == 0):
    #             FP += 1
    #     return FP
    #
    # def Compute_FN(self, test_label, label, fn):
    #     FN = fn
    #     for i in range(len(test_label)):
    #         if(test_label[i] == 0 and label[i] == 1):
    #             FN += 1
    #     return FN

    def Compute_TP_TN_FP_FN(self, test_label, label, matrix):
        matrix = matrix
        for i in range(len(test_label)):
            if test_label[i] == 0 and label[i] == 0:
                matrix[0] += 1
            elif test_label[i] == 1 and label[i] == 1:
                matrix[1] += 1
            elif test_label[i] == 1 and label[i] == 0:
                matrix[2] += 1
            elif test_label[i] == 0 and label[i] == 1:
                matrix[3] += 1
        return matrix
