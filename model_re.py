import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np


# prosize: 5762, plensize:20
# j1:33, s1:1, pf1:64 = window-size, stride-step, No. of filters of first protein-CNN convolution layer
# ja1:17 sa1:1 = window-size, stride-step of first protein-CNN average-pooling layer
# j2:23,s2:1, pf2:64 = second protein-CNN convolution layer
# ja2:11, sa2:1 = second protein-CNN average-pooling layer
# j3:33, s3:1, pf3:32 = third protein-CNN convolution layer
# ja3:17, sa3:1 third protein-CNN average-pooling layer
# n_hid3:70, n_hid4:80, n_hid5:60, n_out:1


class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        # first conv of seq_cnn
        self.conv1_pro = nn.Conv2d(1, 64, (33, 20), stride=(1, ), padding=(33//2, 0))
        self.bn1_pro = nn.BatchNorm2d(64)
        # second conv of seq_cnn
        self.conv2_pro = nn.Conv2d(64, 64, (23, 1), stride=(1, ), padding=(23//2, 0))
        self.bn2_pro = nn.BatchNorm2d(64)
        # third conv of seq_cnn
        self.conv3_pro = nn.Conv2d(64, 32, (33, 1), stride=(1, ), padding=(33//2, 0))
        self.bn3_pro = nn.BatchNorm2d(32)
        fc3_pro = nn.Linear(115240, 70) # 5762 x 20

        self.m1 = (5762+(33//2*2)-33)//1+1
        # print('m1', self.m1)
        self.m2 = (self.m1+(17//2*2)-17)//1+1
        # print('m2', self.m2)
        self.m3 = (self.m2+(23//2*2)-23)//1+1
        # print('m3', self.m3)
        self.m4 = (self.m3+(11//2*2)-11)//1+1
        # print('m4', self.m4)
        self.m5 = (self.m4+(33//2*2)-33)//1+1
        # print('m5', self.m5)
        self.m6 = (self.m5+(17//2*2)-17)//1+1
        # print('m6', self.m6)

    def forward(self, seq):
        seq = self.conv1_pro(seq)  # first conv
        seq = self.bn1_pro(seq)    # batch norm
        seq = F.leaky_relu(seq)    # leaky_relu activation
        seq = F.dropout(seq, p=0.2) # dropout
        seq = F.avg_pool2d(seq, (17, 1), stride=1, padding=(17//2, 0)) # avg_pooling

        seq = self.conv2_pro(seq)
        seq = self.bn2_pro(seq)
        seq = F.leaky_relu(seq)
        seq = F.dropout(seq, p=0.2)
        seq = F.avg_pool2d(seq, (11, 1), stride=1, padding=(11//2, 0))

        seq = self.conv3_pro(seq)
        seq = self.bn3_pro(seq)
        seq = F.leaky_relu(seq)
        seq = F.dropout(seq, p=0.2)
        seq = F.avg_pool2d(seq, (17, 1), stride=1, padding=(17//2, 0))
        seq_protein = F.max_pool2d(seq, (self.m6, 1))
        return seq_protein