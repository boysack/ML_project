from utils import *
from numpy import linalg as LA

def PCA(data):
    C = covariance_matrix(data)

    # compute eigenvalues and eigenvectors (eigh sorts them in ascending order)
    eigenvalues, eigenvectors = LA.eigh(C)

    # project data onto eigenvectors
    projected_data = np.dot(eigenvectors.T, data)

    return projected_data

def LDA(data, labels):
    data = center_data(data)
    data_mean = mean(data)
    classes = classes_number(labels)
    mean_of_classes = compute_mean_of_classes(data, labels)
    S_b = np.zeros((data.shape[0], data.shape[0]))
    S_w = np.zeros((data.shape[0], data.shape[0]))

    for i in range(classes):
        S_b += (labels == i).sum() * np.dot(mean_of_classes[i] - data_mean, (mean_of_classes[i] - data_mean).T)
        S_w += covariance_matrix(data[:, labels == i])  # corresponds to cov matrix because it is (x_c,i - mu)(x_c,i - mu)^T
        #  where x_c,i is the centered data of class i so it corresponds to the sum over classes of centered data that is the covariance matrix

    S_b /= classes
    S_w /= classes

    # compute eigenvalues and eigenvectors (eigh sorts them in ascending order)
    eigenvalues, eigenvectors = LA.eigh(np.dot(LA.inv(S_w), S_b))
    W = eigenvectors[:, ::-1][:, 0:classes - 1]

    # project data onto eigenvectors
    projected_data = np.dot(W.T, data)

    return projected_data
