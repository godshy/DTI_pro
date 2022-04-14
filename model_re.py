import torch
import torch.nn as nn
import numpy as np


# prosize: 5762, plensize:20
# j1:33, s1:1, pf1:64 = window-size, stride-step, No. of filters of first protein-CNN convolution layer
# ja1:17 sa1:1 = window-size, stride-step of first protein-CNN average-pooling layer
# j2:23,s2:1, pf2:64 = second protein-CNN convolution layer
# ja2:11, sa2:1 = second protein-CNN average-pooling layer
# j3:33, s3:1, pf3:32 = third protein-CNN convolution layer
# ja3:17, sa3:1 third protein-CNN average-pooling layer
# n_hid3:70, n_hid4:80, n_hid5:60, n_out:1