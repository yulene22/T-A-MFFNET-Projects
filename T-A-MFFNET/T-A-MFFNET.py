# coding:utf-8
import datetime
import warnings
from Data import *
import torch
from torch import nn
import matplotlib.pyplot as plt
from Index_calculation import *
import torch.nn.functional as F
warnings.filterwarnings("ignore")


epochs = 200
min_acc = 0.6
learning_rate = 0.001
torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes // ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.BN1 = nn.BatchNorm2d(in_planes // ratio)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels=in_planes // ratio, out_channels=in_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.BN2 = nn.BatchNorm2d(in_planes)

        self.sigmoid = hsigmoid()

    def forward(self, x):
        avg_out = self.BN2(self.fc2(self.relu1(self.BN1(self.fc1(self.avg_pool(x))))))
        max_out = self.BN2(self.fc2(self.relu1(self.BN1(self.fc1(self.max_pool(x))))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # avg 和 max 两个描述，叠加 共两个通道。
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)#保持卷积前后H、W不变
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)#通道维度的平均池化
        # 注意 torch.max(x ,dim = 1) 返回最大值和所在索引，是两个值  keepdim = True 保持维度不变（求max的这个维度变为1），不然这个维度没有了
        max_out, _ = torch.max(x, dim=1, keepdim=True)#通道维度的最大池化
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x

input_size = 80  # 输入数据X的特征值的数目。
hidden_size = 80  # 隐藏层的神经元数量，也就是层的特征数
num_layers = 2  # 循环神经网络的层数，默认值是 2
class LSTM_RNN(nn.Module):
    def __init__(self):
        super(LSTM_RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,   # rnn 隐藏单元数
                            num_layers=num_layers,     # rnn 层数
                            batch_first=True,          # 如果设置为True，则输入数据的维度中第一个维度就是 batch 值，默认为False。
                                                       # 默认情况下第一个维度是序列的长度，第二个维度才是 batch?第三个维度是特征数目
                            )

        self.cnn = nn.Sequential(
            nn.Conv2d(17, 64, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

    def forward(self, input_x):

        hidden_cell = (torch.zeros(num_layers, batch_size, hidden_size).to(device),  # shape: (n_layers, batch, hidden_size)
                       torch.zeros(num_layers, batch_size, hidden_size).to(device))

        lstm_out, (hn, cn) = self.lstm(input_x, hidden_cell)
        cnn_out = self.cnn(lstm_out.reshape(batch_size, 17, 5, 16))
        return cnn_out

class Fire(nn.Module):
    def __init__(self, in_channel, out_channel, squzee_channel):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )
        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )
        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)
        return x


class SqueezeNet(nn.Module):
    def __init__(self, class_num=100):
        super().__init__()

        self.ca = ChannelAttention(96)
        self.sa = SpatialAttention()

        self.lstm = LSTM_RNN()
        self.gelu = nn.GELU()

        self.stem = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fire2 = Fire(96, 128, 16)
        self.fire3 = Fire(128, 128, 16)
        self.fire4 = Fire(128, 256, 32)
        self.fire5 = Fire(256, 256, 32)
        self.fire6 = Fire(256, 384, 48)
        self.fire7 = Fire(384, 384, 48)
        self.fire8 = Fire(384, 512, 64)
        self.fire9 = Fire(512, 512, 64)

        self.conv10 = nn.Conv2d(512, class_num, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(1, 2)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):

        lstm_out = self.lstm(x)
        x = self.stem(lstm_out)
        x = self.ca(x) * x
        x = self.sa(x) * x

        f2 = self.fire2(x)
        f3 = self.fire3(f2) + f2
        f4 = self.fire4(f3)
        f4 = self.maxpool(f4)

        f5 = self.fire5(f4) + f4
        f6 = self.fire6(f5)
        f7 = self.fire7(f6) + f6
        f8 = self.fire8(f7)
        f8 = self.maxpool(f8)
        f9 = self.fire9(f8) + f8

        c10 = self.conv10(f9)
        x = self.avg(c10)

        x = x.view(-1, 100)
        x = self.fc(x)
        return x


myModel = SqueezeNet().to(device)
loss_func = nn.MSELoss().to(device)
opt = torch.optim.Adam(myModel.parameters(), lr=learning_rate)

G = testclass()
train_len = G.len(X_train.shape[0], batch_size)
test_len = G.len(X_test.shape[0], batch_size)

train_loss_plt = []
train_acc_plt = []
test_loss_plt = []
test_acc_plt = []
Train_Loss_list = []
Train_Accuracy_list = []
Test_Loss_list = []
Test_Accuracy_list = []
Recall = 0
Precision = 0
F1Score = 0
Specificity = 0

print("开始时间：")
startTime = datetime.datetime.now()
print(startTime)
for i in range(epochs):
    total_train_step = 0
    total_test_step = 0

    total_train_loss = 0
    total_train_acc = 0

    for data in train_dataloader:
        x, y = data
        x = x.to(device)
        y = y.to(device)
        outputs = myModel(x)
        train_loss = loss_func(y, outputs)

        opt.zero_grad()
        train_loss.backward()
        opt.step()

        train_label = G.train_lable2(outputs)
        label = G.train_lable2(y)
        train_acc = G.acc(train_label, label)

        train_loss_plt.append(train_loss)
        total_train_loss = total_train_loss + train_loss.item()
        total_train_step = total_train_step + 1

        train_acc_plt.append(train_acc)
        total_train_acc += train_acc

    Train_Loss_list.append(total_train_loss / (len(train_dataloader)))
    Train_Accuracy_list.append(total_train_acc / train_len)

    total_test_loss = 0
    total_test_acc = 0
    matrix = [0, 0, 0, 0]
    with torch.no_grad():
        pred_output_list = []
        for data in test_dataloader:
            testx, testy = data
            testx = testx.to(device)
            testy = testy.to(device)
            outputs = myModel(testx)
            test_loss = loss_func(testy, outputs)

            test_label = G.train_lable2(outputs)
            label = G.train_lable2(testy)
            test_acc = G.acc(test_label, label)
            TP_TN_FP_FN = G.Compute_TP_TN_FP_FN(test_label, label, matrix)

            test_loss_plt.append(test_loss)
            total_test_loss = total_test_loss + test_loss.item()
            total_test_step = total_test_step + 1

            test_acc_plt.append(test_acc)
            total_test_acc += test_acc

    Test_Loss_list.append(total_test_loss / (len(test_dataloader)))
    Test_Accuracy_list.append(total_test_acc / test_len)

    # 保存准确率高于0.8的模型
    if(total_test_acc / test_len) > min_acc:
        min_acc = total_test_acc / test_len
        res_TP_TN_FP_FN = TP_TN_FP_FN
        torch.save(myModel.state_dict(), 'Sq_L_Se_params.pkl')


    print("Epoch: {}/{} ".format(i + 1, epochs),
          "Training Loss: {:.4f} ".format(total_train_loss / len(train_dataloader)),
          "Training Accuracy: {:.4f} ".format(total_train_acc / train_len),
          "Test Loss: {:.4f} ".format(total_test_loss / len(test_dataloader)),
          "Test Accuracy: {:.4f}".format(total_test_acc / test_len))

endTime = datetime.datetime.now()
print("结束时间：")
print(endTime)
print("总共用时：")
print(endTime - startTime)
print(min_acc)
print("TP: {}".format(res_TP_TN_FP_FN[0]))
print("TN: {}".format(res_TP_TN_FP_FN[1]))
print("FP: {}".format(res_TP_TN_FP_FN[2]))
print("FN: {}".format(res_TP_TN_FP_FN[3]))
print("Recall: {}".format(res_TP_TN_FP_FN[0] / (res_TP_TN_FP_FN[0] + res_TP_TN_FP_FN[3])))
print("Precision: {}".format(res_TP_TN_FP_FN[0] / (res_TP_TN_FP_FN[0] + res_TP_TN_FP_FN[2])))
print("F1Score: {}".format(
    2 * (res_TP_TN_FP_FN[0] / (res_TP_TN_FP_FN[0] + res_TP_TN_FP_FN[2]))
      * (res_TP_TN_FP_FN[0] / (res_TP_TN_FP_FN[0] + res_TP_TN_FP_FN[3]))
      / ((res_TP_TN_FP_FN[0] / (res_TP_TN_FP_FN[0] + res_TP_TN_FP_FN[2]))
      + (res_TP_TN_FP_FN[0] / (res_TP_TN_FP_FN[0] + res_TP_TN_FP_FN[3])))
))
print("Specificity: {}".format(res_TP_TN_FP_FN[1] / (res_TP_TN_FP_FN[1] + res_TP_TN_FP_FN[2])))

train_x1 = range(0, 200)
train_x2 = range(0, 200)
train_y1 = Train_Accuracy_list
train_y2 = Train_Loss_list
plt.subplot(2, 1, 1)
plt.plot(train_x1, train_y1, 'o-')
plt.title('Train accuracy vs. epoches')
plt.ylabel('Train accuracy')
plt.subplot(2, 1, 2)
plt.plot(train_x2, train_y2, '.-')
plt.xlabel('Train loss vs. epoches')
plt.ylabel('Train loss')
plt.show()
plt.savefig("accuracy_loss.jpg")

test_x1 = range(0, 200)
test_x2 = range(0, 200)
test_y1 = Test_Accuracy_list
test_y2 = Test_Loss_list
plt.subplot(2, 1, 1)
plt.plot(test_x1, test_y1, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(test_x2, test_y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()
plt.savefig("accuracy_loss.jpg")










