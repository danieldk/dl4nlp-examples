import numpy as np

""" Generates n gaussians with the given mean and standard deviation."""
def gaussian_around(means, n, stddev=0.1):
    cov = np.identity(means.shape[0]) * (stddev ** 2)
    return np.random.multivariate_normal(means, cov, n)
    
""" Generates random data points from the set of classes."""
def gaussians(means, labels, n, stddev=0.1):
    (n_labels, n_dims) = means.shape
    
    for i in range(n):
        idx = np.random.randint(0, n_labels)
        yield gaussian_around(means[idx], 1, stddev), labels[idx]

"""Generate n random data points from the given labels with the given means."""
def n_gaussians(means, labels, n, stddev=0.1):
    (n_labels, n_dims) = means.shape
        
    X = np.zeros((n, n_dims))
    Y = np.zeros(n, dtype=np.int32)
    
    for idx, (x, y) in enumerate(gaussians(means, labels, n, stddev)):
        X[idx] = x
        Y[idx] = y
    
    return X, Y
