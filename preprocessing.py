from utils import *
from numpy import linalg as LA
from abc import abstractmethod

class prepocess:

    def __init__(self, data, m_control_value:int, m:int) -> None:
        if(m>m_control_value):
            self.m = m_control_value
        else:
            self.m = m
        self.data = data

    @abstractmethod
    def process(self):
        pass

class pca(prepocess):

    def __init__(self, data, m:int=1) -> None:
        super().__init__(data, data.shape[0], m)
        self.eigenvalues = None
        self.eigenvectors = None

    def process(self):
        C = covariance_matrix(self.data)
        self.eigenvalues, self.eigenvectors = LA.eigh(C)
        self.data = np.dot(self.eigenvectors.T, self.data)

class lda(prepocess):
    def __init__(self, data, labels, m:int=1) -> None:
        super().__init__(data, np.unique(labels).size-1, m)
        self.labels = labels
        self.eigenvalues = None
        self.eigenvectors = None

    def process(self):
        data_mean = mean(self.data)
        classes = classes_number(self.labels)
        mean_of_classes = compute_mean_of_classes(self.data, self.labels)
        S_b = np.zeros((self.data.shape[0], self.data.shape[0]))
        S_w = np.zeros((self.data.shape[0], self.data.shape[0]))
        for i in range(classes):
            S_b += (self.labels == i).sum() * np.dot(mean_of_classes[i] - data_mean, (mean_of_classes[i] - data_mean).T)
            S_w += np.dot(self.data[:, self.labels == i] - mean_of_classes[i], (self.data[:, self.labels == i] - mean_of_classes[i]).T)
        S_b /= self.data.shape[1]
        S_w /= self.data.shape[1]

        # compute eigenvalues and eigenvectors (eigh sorts them in ascending order)
        _, eigenvectors = LA.eigh(np.dot(LA.inv(S_w), S_b))
        W = eigenvectors[:, ::-1][:, 0:classes - 1]
        # project data onto eigenvectors
        self.data = np.dot(W.T, self.data)