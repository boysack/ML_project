from abc import abstractmethod
import numpy as np
import scipy.special as sp
import scipy.optimize as opt
from utils import *

class gaussian_mixture_model:
    def __init__(self, data:np.ndarray, labels:np.ndarray, data_test:np.ndarray, ) -> None:
        self.data = data
        self.data_test = data_test
        self.labels = labels
        self.means = None
        self.covariances = None
        self.weights = None
        self.score_values = None
        self.is_fitted = False
        self.logdens = None

    def logpdf_GMM(self, data:np.ndarray, params:np.ndarray): #  params = [weights, means, covariances]
        # params = [(w1, mu1, C1), (w2, mu2, C2), ...]
        score_matrix = np.zeros((len(params), data.shape[1])) #  each row contains the score of a gaussian and in each row, the column corresponds to the score of that gaussian for the j-th feature of the i-th sample

        # S_0,0 = score della prima gaussiana per la prima feature
        # S_0,1 = score della prima gaussiana per la seconda feature
        # S_1,0 = score della seconda gaussiana per la prima feature
        # S_1,1 = score della seconda gaussiana per la seconda feature

        for element in range(data.shape[1]): #  for each element
            for (index, parameter) in enumerate(params):
                # prova = logpdf_GAU_ND(data[:, element], parameter[1], parameter[2])
                score_matrix[index, element] = gaussian_log_pdf(data[:, element:element+1], parameter[1], parameter[2]) + np.log(parameter[0])
                

        self.logdens = sp.logsumexp(score_matrix, axis=0)

def gaussian_log_pdf(X:np.ndarray, mu:np.ndarray, C:np.ndarray): #  claudio
    inv_C = np.linalg.inv(C)

    # c1 = -0.5 * C.size * np.log(2*np.pi)
    c1 = -0.5 * C.shape[0] * np.log(2*np.pi)
    logdet = -0.5 * np.linalg.slogdet(C)[1]
    s = -0.5 * ((X - mu) * np.dot(inv_C, (X - mu))).sum(0)
    result = c1 + logdet + s
    
    # print(result)
    return result

def logpdf_GAU_ND(X: np.ndarray, mu: np.ndarray, C: np.ndarray) -> np.ndarray: #  ragazzi
    M = X.shape[0]
    _, det_C = np.linalg.slogdet(C)
    inv_C = np.linalg.inv(C)
    density_array = -0.5 * M * np.log(2 * np.pi) - 0.5 * det_C
    density_array = density_array - 0.5 * ((X - mu) * np.dot(inv_C, (X - mu))).sum(0)

    # print(density_array)
    return density_array




        

        

