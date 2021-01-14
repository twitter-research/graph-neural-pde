import os
import h5py
import numpy as np
import scipy.sparse as sp


# Auxiliary Functions
def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir   = np.asarray(ds['ir'])
            jc   = np.asarray(ds['jc'])
            out  = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()
    return out

def stack_matrices(x, n_supPix): # stack 2D matrices into a 3D tensor
    x_rho = x[:,0:n_supPix*n_supPix]
    x_theta = x[:, n_supPix*n_supPix:]
    x_rho = np.reshape(x_rho, (x_rho.shape[0], int(np.sqrt(x_rho.shape[1])), int(np.sqrt(x_rho.shape[1])), 1))
    x_theta = np.reshape(x_theta, (x_theta.shape[0], int(np.sqrt(x_theta.shape[1])), int(np.sqrt(x_theta.shape[1])), 1))
    for k in range(x_theta.shape[0]): #set the diagonal of all the theta matrix to a value really close to 0
        np.fill_diagonal(x_theta[k,:,:,0], 1e-14)
    y = np.concatenate([x_rho, x_theta], axis=3)
    return y

def compute_similarity_matrix(dist_matrix):
    shp = dist_matrix.shape
    similarity_matrix = np.zeros(shp)
    sigma = np.mean(dist_matrix[np.isfinite(dist_matrix)])
    for i in range(shp[0]):
        for j in range(shp[1]):
            if (np.isfinite(dist_matrix[i,j])):
                dist = np.exp(-dist_matrix[i,j]**2/sigma**2) #the higher the distance the smaller the similarity
                similarity_matrix[i, j] = dist
            else:
                similarity_matrix[i, j] = 0
    return similarity_matrix


class Dataset(object):
    def __init__(self, path_train_vals, path_test_vals, path_coords_train,
                 path_coords_test, path_train_labels, path_test_labels,
                 n_supPix):

        # path to (pre-computed) descriptors
        self.path_train_vals = path_train_vals
        self.path_test_vals = path_test_vals

        # path to (pre-computed) patches
        self.path_coords_train = path_coords_train
        self.path_coords_test = path_coords_test

        # path to labels
        self.path_train_labels = path_train_labels
        self.path_test_labels = path_test_labels

        # loading the descriptors
        print("[i] Loading signals")
        self.vals_train = load_matlab_file(self.path_train_vals, 'vals')
        self.vals_test = load_matlab_file(self.path_test_vals, 'vals')

        # loading the coords
        print("[i] Loading coords")
        tmp = load_matlab_file(self.path_coords_train, 'patch_coords')
        self.coords_train = stack_matrices(tmp, n_supPix)

        tmp = load_matlab_file(self.path_coords_test, 'patch_coords')
        self.coords_test = stack_matrices(tmp, n_supPix)

        # compute the adjacency matrix
        self.adj_mat_train = np.zeros(
            (self.coords_train.shape[0], self.coords_train.shape[1], self.coords_train.shape[2]))
        for k in range(self.coords_train.shape[0]):
            self.adj_mat_train[k, :, :] = np.isfinite(self.coords_train[k, :, :, 1])

        self.adj_mat_test = np.zeros((self.coords_test.shape[0], self.coords_test.shape[1], self.coords_test.shape[2]))
        for k in range(self.coords_test.shape[0]):
            self.adj_mat_test[k, :, :] = np.isfinite(self.coords_test[k, :, :, 1])

        print("[i] Loading labels")
        self.train_labels = self.load_labels(self.path_train_labels)
        self.test_labels = self.load_labels(self.path_test_labels)

    def load_labels(self, fname):
        tmp = load_matlab_file(fname, 'labels')
        tmp = tmp.astype(np.int32)
        return tmp.flatten()

if __name__ == "__main__":
    # Data Loading
    # path to the main folder
    n_supPix = 75
    path_main = '../data/SuperMNIST/MNIST/'
    # path to the input descriptors
    path_train_vals    = os.path.join(path_main,'datasets/mnist_superpixels_data_%d/train_vals.mat' % n_supPix)
    path_test_vals    = os.path.join(path_main,'datasets/mnist_superpixels_data_%d/test_vals.mat' % n_supPix)
    # path to the patches
    path_coords_train  = os.path.join(path_main,'datasets/mnist_superpixels_data_%d/train_patch_coords.mat' % n_supPix)
    path_coords_test  = os.path.join(path_main,'datasets/mnist_superpixels_data_%d/test_patch_coords.mat' % n_supPix)
    # path to the labels
    path_train_labels   = os.path.join(path_main,'datasets/MNIST_preproc_train_labels/MNIST_labels.mat')
    path_test_labels   = os.path.join(path_main,'datasets/MNIST_preproc_test_labels/MNIST_labels.mat')
    # path to the idx centroids
    path_train_centroids  = os.path.join(path_main,'datasets/mnist_superpixels_data_%d/train_centroids.mat' % n_supPix)
    path_test_centroids  = os.path.join(path_main,'datasets/mnist_superpixels_data_%d/test_centroids.mat' % n_supPix)

    ds = Dataset(path_train_vals, path_test_vals, path_coords_train, path_coords_test, path_train_labels,
                     path_test_labels, n_supPix)