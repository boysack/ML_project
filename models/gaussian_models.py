from abc import abstractmethod
import numpy as np
from utils import *
from utils import np

class gaussian:
  def __init__(self, data:np.ndarray, labels:np.ndarray):
    self.data = data
    self.labels = labels
    self.means = None
    self.covariances = None
    self.is_fitted = False

  @abstractmethod
  def fit():
    pass

  @abstractmethod
  def predict():
    pass
  
  def gaussian_log_pdf(X:np.ndarray, C:np.ndarray, mu:np.ndarray):
    inv_C = np.linalg.inv(C)

    c1 = -0.5 * C.size * np.log(2*np.pi)
    logdet = -0.5 * np.linalg.slogdet(C)[1]
    s = -0.5 * ((X - mu) * np.dot(inv_C, (X - mu))).sum(0)

    return c1 + logdet + s

  def binary_gaussian_score(self, X:np.ndarray):
    lr = np.exp(self.gaussian_log_pdf(X,self.C[1],self.mu[1])-self.gaussian_log_pdf(X,self.C[0],self.mu[0]))
    return lr
  
  def predict(self, X:np.ndarray):
    if(self.is_fitted is False):
      self.fit()
    return self.binary_gaussian_score()

class mvg(gaussian):
  def __init__(self, data:np.ndarray, labels:np.ndarray):
    super().__init__(data, labels)

  def fit(self):
    #v_col(np.array(compute_mean_of_classes(self.data))) ?????
    self.means = compute_mean_of_classes(self.data, self.labels)
    self.covariances = compute_covariance_of_classes(self.data, self.labels)
    self.is_fitted = True
    
class naiveg(gaussian):
  def __init__(self, data:np.ndarray, labels:np.ndarray):
    super().__init__(data, labels)

  def fit(self):
    self.means = compute_mean_of_classes(self.data, self.labels)
    covariances = compute_covariance_of_classes(self.data, self.labels)
    dim = self.data.shape[0]
    classes = classes_number(self.labels)
    self.covariances = [covariances[i] * np.identity(dim) for i in range(classes)]
    self.is_fitted = True

class tiedg(gaussian):
  def __init__(self, data:np.ndarray, labels:np.ndarray):
    super().__init__(data, labels)

  def fit(self):
    self.means = compute_mean_of_classes(self.data, self.labels)
    self.covariances = covariance_matrix(self.data)

  def binary_gaussian_score(self, X:np.ndarray):
    lr = np.exp(self.gaussian_log_pdf(X,self.C,self.mu[1])-self.gaussian_log_pdf(X,self.C,self.mu[0]))
    return lr