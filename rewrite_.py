import gc

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

#  rewrite and tear apart down each section


def seq_cnn(seq, plen_size, pro_size):
    pf1 = 64
    pf2 = 64
    pf3 = 32
    j1 = 33
    j2 = 23
    j3 = 33
    s1 = 1
    s2 = 1
    s3 = 1
    ja1 = 17
    ja2 = 11
    ja3 = 17
    sa1 = 1
    sa2 = 1
    sa3 = 1
    n_hid3 = 70
    conv1_pro = nn.Conv2d(1, pf1, (j1, plen_size), stride=s1, padding=(int(j1//2), 0), padding_mode='zeros')
    conv2_pro = nn.Conv2d(pf1, pf2, (j2, 1), stride=s2, padding=(int(j2//2), 0), padding_mode='zeros')
    bn1_pro = nn.BatchNorm2d(pf1)
    bn2_pro = nn.BatchNorm2d(pf2)
    conv3_pro = nn.Conv2d(pf2, pf3, (j3, 1), stride=s3, padding=(int(j3//2), 0), padding_mode='zeros')
    bn3_pro = nn.BatchNorm2d(pf3)

    m1 = (pro_size+(j1//2*2)-j1)//s1+1
    m2 = (m1+(ja1//2*2)-ja1)//sa1+1
    m3 = (m2+(j2//2*2)-j2)//s2+1
    m4 = (m3+(ja2//2*2)-ja2)//sa2+1
    m5 = (m4+(j3//2*2)-j3)//s3+1
    m6 = (m5+(ja3//2*2)-ja3)//sa3+1
    # problem
    print(type(seq))
    seq = torch.from_numpy(seq.astype(np.float32)).clone()
    print(type(seq))
    output = conv1_pro(seq)
    del seq
    gc.collect()
    # problem
    bn_output = bn1_pro(output)
    del output
    gc.collect()
    print('batch normalization after ')
    h = F.dropout(F.leaky_relu(bn_output), p=0.2)  # 1st conv
    print('end of 1st conv')
    h = F.avg_pool2d(h, (ja1, 1), stride=sa1, padding=(ja1//2, 0))  # 1st pooling
    h = F.dropout(F.leaky_relu(bn2_pro(conv2_pro(h))), p=0.2)  # 2nd conv
    h = F.avg_pool2d(h, (ja2, 1), stride=sa2, padding=(ja2//2, 0))  # 2nd pooling
    h = F.dropout(F.leaky_relu(bn3_pro(conv3_pro(h))), p=0.2)  # 3rd conv
    h = F.avg_pool2d(h, (ja3, 1), stride=sa3, padding=(ja3//2, 0))  # 3rd pooling
    h_pro = F.max_pool2d(h, (m6, 1))  # global max pooling, fingerprint
    print('h_pro_shape:', h_pro.shape)
    return h_pro
    fc3_pro = nn.Linear(3, n_hid3)
    h_pro = F.dropout(F.leaky_relu(fc3_pro(h_pro)), p=0.2) # fully connected_1
    #print(h_pro.shape)
    # return h_pro

feature_vector_seq = 20
pro_size = 5762
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print('use'+device)


file_sequences = np.load('./dataset_hard'+'/cv_'+str(0)+'/train_reprotein.npy')
print('Loading sequences: train_reprotein.npy', flush=True)
sequences = np.asarray(file_sequences, dtype='float32').reshape((-1, 1, pro_size, feature_vector_seq))
# type: numpy_array ---- torch.tensor
seq_cnn(sequences, feature_vector_seq, pro_size).to(device)

print('OVER')