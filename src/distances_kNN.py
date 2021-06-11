import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree, DistanceMetric


def apply_feat_KNN(x, k):
  nbrs = NearestNeighbors(n_neighbors=k).fit(x)
  distances, indices = nbrs.kneighbors(x)
  src = np.linspace(0, len(x) * k, len(x) * k + 1)[:-1] // k
  dst = indices.reshape(-1)
  ei = np.vstack((src, dst))
  return ei

def apply_dist_KNN(x, k):
  nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(x)
  distances, indices = nbrs.kneighbors(x)
  src = np.linspace(0, len(x) * k, len(x) * k + 1)[:-1] // k
  dst = indices.reshape(-1)
  ei = np.vstack((src, dst))
  return ei

def threshold_mat(dist, quant=1/1000):
  thresh = np.quantile(dist, quant, axis=None)
  A = dist <= thresh
  return A

def make_ei(A):
  src, dst = np.where(A)
  ei = np.vstack((src, dst))
  return ei

def apply_dist_threshold(dist, quant=1/1000):
  return make_ei(threshold_mat(dist, quant))


def get_distances(x):
  dist = DistanceMetric.get_metric('euclidean')
  return dist.pairwise(x)

if __name__ == "__main__":
  # triangele
  # dist = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
  # square
  dist = np.array([[0, 1, 1, np.sqrt(2)], [1, 0, np.sqrt(2), 1], [1, np.sqrt(2), 0, 1], [np.sqrt(2), 1, 1, 0]])
  print(f"distances \n {dist}")

  for k in range(4):  # 3
    print(f"{k + 1} edges \n {apply_dist_KNN(dist, k + 1)}")

  quant= 0.75
  thresh = np.quantile(dist, quant, axis=None)

  A = threshold_mat(dist, quant)
  print(f"Threshold mat \n {A}")
  print(f"Edge index1 \n {make_ei(A)}")
  print(f"Edge index2 \n {apply_dist_threshold(dist, quant)}")

  square = np.array([[0,1],[1,1],[0,0],[1,0]])
  sq_dist = get_distances(square)
  print(f"sq_dist \n {sq_dist}")