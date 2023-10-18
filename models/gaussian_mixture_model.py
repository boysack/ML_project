from abc import abstractmethod
import numpy as np
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

    def logpdf_GMM(self, data:np.ndarray, params:np.ndarray): #  params = [weights, means, covariances]
        score_matrix = np.zeros((len(params), data.shape[1])) #  each row contains the score of a gaussian and in each row, the column corresponds to the score of that gaussian for the j-th feature of the i-th sample

        # S_0,0 = score della prima gaussiana per la prima feature
        # S_0,1 = score della prima gaussiana per la seconda feature
        # S_1,0 = score della seconda gaussiana per la prima feature
        # S_1,1 = score della seconda gaussiana per la seconda feature
        for feature in range(data.shape[1]):
            for param in params: #  each param is a gaussian (weights, means, covariances)
                score_matrix[params.index(param), feature] = self.gaussian_log_pdf(data[:, feature], param[1][:, feature], param[2][:, feature])
                # riga 26 da rivedere, aggiungere il parametro weights

        # return logdens = scipy.special.logsumexp(S, axis=0)



    def gaussian_log_pdf(self, X:np.ndarray, mu:np.ndarray, C:np.ndarray): #  claudio
        inv_C = np.linalg.inv(C)

        c1 = -0.5 * C.size * np.log(2*np.pi)
        logdet = -0.5 * np.linalg.slogdet(C)[1]
        s = -0.5 * ((X - mu) * np.dot(inv_C, (X - mu))).sum(0)

        return c1 + logdet + s
        

        

