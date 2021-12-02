import ignite.metrics
import keras.models
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics import Accuracy, Loss
import torch.optim as optim
import numpy as np
import pickle
import os



# prosize, plensize_20 = size of protein one hot feature matrix
# j1, s1, pf1 = window-size, stride-step, No. of filters of first protein-CNN convolution layer
# ja1, sa1 = window-size, stride-step of first protein-CNN average-pooling layer
# j2, s2, pf2 = window-size, stride-step, No. of filters of second protein-CNN convolution layer
# ja2, sa2 = window-size, stride-step of second protein-CNN average-pooling layer
# j3, s3, pf3 = window-size, stride-step, No. of filters of third protein-CNN convolution layer
# ja3, sa3 = window-size, stride-step of third protein-CNN average-pooling layer


class DeepCNN(nn.Module):
    def __init__(self, prosize, plensize, batchsize, s1, sa1, s2, sa2, s3, sa3, j1, pf1, ja1, j2, pf2, ja2, j3, pf3, ja3, n_hid3, n_hid4, n_hid5, n_out, *args, **kwargs):
        super(DeepCNN, self).__init__()
        self.conv1_pro=nn.Conv2d(1, pf1, (j1, plensize), stride=s1, padding=(j1//2, 0)),
        # conv1_pro=nn.Conv2d(1, pf1, (j1, plensize), stride=s1, padding=(j1//2, 0)),
        self.bn1_pro=nn.BatchNorm2d(pf1),
        self.conv2_pro=nn.Conv2d(pf1, pf2, (j2, 1), stride=s2, padding=(j2//2, 0)),
        self.bn2_pro=nn.BatchNorm2d(pf2),
        self.conv3_pro=nn.Conv2d(pf2, pf3, (j3, 1), stride=s3, padding=(j3//2, 0)),
        self.bn3_pro=nn.BatchNorm2d(pf3),
        self.fc4=nn.Linear(1, n_hid4),
        self.fc5=nn.Linear(2, n_hid5),
        self.fc3_pro=nn.Linear(3, n_hid3),
        self.fc4_pro=nn.Linear(4, n_hid4),
        self.fc5_pro=nn.Linear(5, n_hid5),
        self.fc6=nn.Linear(6, n_out)

        self.n_hid3, self.n_hid4, self.n_hid5, self.n_out = n_hid3, n_hid4, n_hid5, n_out
        self.prosize, self.plensize = prosize, plensize
        self.s1, self.sa1, self.s2, self.sa2, self.s3, self.sa3 = s1, sa1, s2, sa2, s3, sa3
        self.j1, self.ja1, self.j2, self.ja2, self.j3, self.ja3 = j1, ja1, j2, ja2, j3, ja3

        self.m1 = (self.prosize+(self.j1//2*2)-self.j1)//self.s1+1
        self.m2 = (self.m1+(self.ja1//2*2)-self.ja1)//self.sa1+1
        self.m3 = (self.m2+(self.j2//2*2)-self.j2)//self.s2+1
        self.m4 = (self.m3+(self.ja2//2*2)-self.ja2)//self.sa2+1
        self.m5 = (self.m4+(self.j3//2*2)-self.j3)//self.s3+1
        self.m6 = (self.m5+(self.ja3//2*2)-self.ja3)//self.sa3+1

    def __call__(self, ecfp, sequences, n2vc, n2vp, interactions):
        z = self.cos_similarity(ecfp, sequences, n2vc, n2vp)
        Z = self.fc6(z)
        loss = F.cosine_similarity(Z, interactions)
        # loss = tf.compat.v1.losses.sigmoid_cross_entropy(Z, interactions)
        # ---------------------------------------------------------------
        accuracy = ignite.metrics.Accuracy(Z, interactions)
        # accuracy_ = tf.keras.metrics.binary_accuracy(Z, interactions) #---
        # ---------------------------------------------------------------
        print({'loss': loss, 'accuracy': accuracy}, self)
        return loss

    def predict_pro(self, seq):
        h = F.dropout(F.leaky_relu(self.bn1_pro(self.conv1_pro(seq))), p=0.2)  # 1st conv
        h = F.avg_pool2d(h, (self.ja1,1), stride=self.sa1, pad=(self.ja1//2, 0))  # 1st pooling
        h = F.dropout(F.leaky_relu(self.bn2_pro(self.conv2_pro(h))), p=0.2)  # 2nd conv
        h = F.avg_pool2d(h, (self.ja2,1), stride=self.sa2, pad=(self.ja2//2, 0))  # 2nd pooling
        h = F.dropout(F.leaky_relu(self.bn3_pro(self.conv3_pro(h))), p=0.2)  # 3rd conv
        h = F.avg_pool2d(h, (self.ja3,1), stride=self.sa3, pad=(self.ja3//2, 0))  # 3rd pooling
        h_pro = F.max_pool2d(h, (self.m6,1)) # global max pooling, fingerprint
        #print(h_pro.shape)
        h_pro = F.dropout(F.leaky_relu(self.fc3_pro(h_pro)), p=0.2)# fully connected_1
        #print(h_pro.shape)
        return h_pro

    def cos_similarity(self, fp, seq, n2c, n2p):
        x_compound = fp
        x_protein = self.predict_pro(seq)
        x_compound = self.fc4(torch.cat((x_compound, n2c)))
        x_compound = F.dropout(F.leaky_relu(x_compound), p=0.2)
        x_compound = F.dropout(F.leaky_relu(self.fc5(x_compound)), p=0.2)
        x_protein = self.fc4_pro(torch.cat((x_protein, n2p)))
        x_protein = F.dropout(F.leaky_relu(x_protein), p=0.2)
        #print(x_protein.shape)
        x_protein = F.dropout(F.leaky_relu(self.fc5_pro(x_protein)), p=0.2)
        #print(x_protein.shape)
        y = x_compound * x_protein
        return y


    #  conv1_pro=L.Convolution2D(1, pf1, (j1, plensize), stride=s1, pad = (j1//2,0)),

''' 
def model(seq):
    optimizer = SGD(lr=0.01, momentum=0.9, decay=0.00001)

      self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
      self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
      self.aux_output = tf.keras.layers.Dense
      def call(self, inputs):
      input_A, input_B = inputs
      hidden1 = self.hidden1(input_B)
      hidden2 = self.hidden2(hidden1)
      concat = tf.keras.layers.concatenate([input_A, hidden2])
      main_output = self.main_output(concat)
      aux_out = self.aux_output(hidden2)
      return main_output, aux_out
'''