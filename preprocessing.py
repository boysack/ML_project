from utils import *
from numpy import linalg as LA
import scipy as sp

def PCA(data:np.ndarray) -> np.ndarray:
    C = covariance_matrix(data)

    # compute eigenvalues and eigenvectors (eigh sorts them in ascending order)
    _, eigenvectors = LA.eigh(C)

    # project data onto eigenvectors
    projected_data = np.dot(eigenvectors.T, data)

    return projected_data

def LDA(data:np.ndarray, labels:np.ndarray) -> np.ndarray:
    data_mean = mean(data)
    classes = classes_number(labels)
    mean_of_classes = compute_mean_of_classes(data, labels)
    S_b = np.zeros((data.shape[0], data.shape[0]))
    S_w = np.zeros((data.shape[0], data.shape[0]))

    for i in range(classes):
        S_b += (labels == i).sum() * np.dot(mean_of_classes[i] - data_mean, (mean_of_classes[i] - data_mean).T)
        S_w += np.dot(data[:, labels == i] - mean_of_classes[i], (data[:, labels == i] - mean_of_classes[i]).T)

    S_b /= data.shape[1]
    S_w /= data.shape[1]

    # compute eigenvalues and eigenvectors (eigh sorts them in ascending order)
    # _, eigenvectors = LA.eigh(np.dot(LA.inv(S_w), S_b))
    _, eigenvectors = sp.linalg.eigh(S_b, S_w)
    # _, eigenvectors = LA.eigh(S_b, S_w)
    W = eigenvectors[:, ::-1][:, 0:classes - 1]

    # project data onto eigenvectors
    projected_data = np.dot(W.T, data)

    return projected_data
