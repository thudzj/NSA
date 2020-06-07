import numpy as np

num_nodes = 8
arch_p = 0.7
results = []
for i in range(10000, 20000):
    rng = np.random.RandomState(i)
    tmp = []
    for j in range(3):
        adj_one = np.tril(rng.rand(num_nodes+1, num_nodes+1) > arch_p, -1).astype(np.int) + np.eye(num_nodes+1).astype(np.int)
        tmp.append(adj_one.reshape(-1))
    results.append(np.concatenate(tmp))
results = np.stack(results).astype(np.int8)
print(results.shape)
equal = (np.abs(results[:, None, :] - results[None, :, :]).sum(2) == 0).astype(np.int)
print(equal[0][:10])
equal = np.tril(equal, -1)
print(equal.sum())