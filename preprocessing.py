from utils import *
from abc import abstractmethod
import scipy as sp

class prepocess:
    def __init__(self, data) -> None:
        self.original_data = data
        self.m = None
        self.projected_data = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.is_preprocessed = False

    @abstractmethod
    def compute_directions(self):
        pass

    @abstractmethod
    def process(self):
        pass

class pca(prepocess):
    def __init__(self, data) -> None:
        super().__init__(data)

    def compute_directions(self):
        C = covariance_matrix(self.original_data)
        self.eigenvalues, self.eigenvectors = sp.linalg.eigh(C)
        # invert order
        self.eigenvalues = self.eigenvalues[::-1]
        self.eigenvectors = self.eigenvectors[::-1]
        # print("eigenvectors: ", self.eigenvalues.shape)
        # self.eigenvectors = self.eigenvectors[:self.m]

    def process(self, m:int=1):
        if(self.eigenvectors is None):
            self.compute_directions()
        if(self.m is None or self.m != m):
            control_value = self.original_data.shape[0]
            if(m<1 and m>control_value):
                self.m = 1
            self.projected_data = np.dot(self.eigenvectors[:, :self.m].T, self.original_data)
            self.is_preprocessed = True

#binary lda - m = 1
class lda(prepocess):
    def __init__(self, data, labels) -> None:
        super().__init__(data)
        self.labels = labels
        self.m = 2

    def compute_directions(self):
        data_mean = mean(self.original_data)
        classes = classes_number(self.labels)
        mean_of_classes = compute_mean_of_classes(self.original_data, self.labels)
        S_b = 0
        S_w = 0
        for i in range(classes):
            S_b += (self.labels == i).sum() * np.dot(mean_of_classes[i] - data_mean, (mean_of_classes[i] - data_mean).T)
            S_w += np.dot(self.original_data[:, self.labels == i] - mean_of_classes[i], (self.original_data[:, self.labels == i] - mean_of_classes[i]).T)
        S_b /= self.original_data.shape[1]
        S_w /= self.original_data.shape[1]

        # compute eigenvalues and eigenvectors (eigh sorts them in ascending order)
        self.eigenvalues, self.eigenvectors = sp.linalg.eigh(S_b, S_w)
        # invert order
        self.eigenvalues = self.eigenvalues[::-1]
        self.eigenvectors = self.eigenvectors[:, ::-1]

    def process(self):
        if(self.eigenvectors is None):
            self.compute_directions()
        self.projected_data = np.dot(self.eigenvectors[:,:self.m].T, self.original_data)
        self.is_preprocessed = True