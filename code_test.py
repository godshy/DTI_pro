import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
data = iris['data']
# print('Input data (N, featuers):', data, data.shape)
N = data.shape[0]
n_dim = data.shape[1]
cov = np.zeros((n_dim, n_dim))

# E(X)
avg_1 = np.mean(data, axis=0)
print('avg_1', avg_1)

avg = []
for i in range(n_dim):
    avg.append(sum([d[i] for d in data]) / N)
print('avg', avg)

# covariance matrix
for i in range(n_dim):
    for j in range(n_dim):
        var = 0
        for d in data:
            var += (d[i] - avg[i]) * (d[j] - avg[j])
        var /= (N - 1)
        print(var)
        cov[i, j] = var

print('cov', cov)
print('sum cov: ', np.sum(cov))

print('numpy cov', np.cov(data, rowvar=False, bias=False))
print('sum numpy cov: ', np.sum(np.cov(data, rowvar=False, bias=False)))