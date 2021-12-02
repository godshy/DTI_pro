import chainer.links as L
import numpy as np

x = np.arange(12).reshape(4, 3).astype(np.float32) ** 2
bn = L.BatchNormalization(3)
print(bn(x))
