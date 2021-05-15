"""
matrix factorisation of the positional encoding required for arxiv-ogbn
"""

import numpy as np

X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
X = np.random.rand(10, 10)
from sklearn.decomposition import NMF

model = NMF(n_components=3, init='random', random_state=0, max_iter=1000)
W = model.fit_transform(X)
# H = model.components_

np.save(W)
