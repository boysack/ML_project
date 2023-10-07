from abc import abstractmethod
import numpy as np
from utils import *

from scipy.optimize import fmin_l_bfgs_b

# implementare kernel e tutto in un dictionary di funzioni
def polynomial_kernel(X_1:np.ndarray, X_2:np.ndarray, k:float, c:float, d:float):
  return (np.dot(X_1.T, X_2) + c) ** d + k ** 2

def radial_basis_function_kernel():
  pass

class svm:
  def __init__(self, data:np.ndarray, labels:np.ndarray, k:float, C:float) -> None:
        self.data = data
        self.labels = labels
        self.k = k
        self.C = C

        self.z = 2*labels-1

        self.alpha = None
        self.weights = None
        self.bias = None
        self.is_fitted = False
  
  @abstractmethod
  def H():
    pass

  # wrapper?
  def obj(self, alpha:np.ndarray) -> np.ndarray:
    H_hat = self.H()
    obj_f = .5 * np.dot(np.dot(alpha.T, H_hat), alpha) - np.dot(alpha.T, v_col(np.ones((self.data.shape[1]))))
    gradient_f = v_col(np.dot(H_hat, alpha))-v_col(np.ones(self.data.shape[1]))
    return obj_f, gradient_f
  
  def train(self):
    alpha = np.zeros((self.data.shape[1], 1))
    box_const = (0, self.C)
    alpha, _, _ = fmin_l_bfgs_b(self.obj, alpha, factr=1.0, bounds=[box_const] * self.data.shape[1])
    self.alpha = alpha
  
    self.is_fitted = True

  @abstractmethod
  def scores():
    pass

class hard_svm(svm):
  def __init__(self, data:np.ndarray, labels:np.ndarray, k:float, C:float):
    super().__init__(data, labels, k, C)

  # forse in superclasse?
  # forse si può reimplementare solo questa parte per implementare i kernel
  def H(self) -> np.ndarray:
    X_hat = np.vstack((self.data, [self.k] * self.data.shape[1]))
    X_hat = X_hat * self.z
    return np.dot(X_hat.T, X_hat)
  
  # forse in superclasse?
  def alpha_to_w_b(self, alpha:np.ndarray) -> np.ndarray:
    X_hat = np.vstack((self.data, [self.k] * self.data.shape[1]))
    return np.sum((v_col(alpha) * v_col(self.z)).T * X_hat, axis=1)

  # forse si può reimplementare solo questa parte per implementare i kernel
  def scores(self, X:np.ndarray):
    if(self.is_fitted is False):
      self.train()
    w_hat = self.alpha_to_w_b(self.alpha)
    weights = w_hat[:-1]
    bias = w_hat[-1]
    return np.dot(weights.T, X) + bias * self.k

class polynomial_svm(svm):
  def __init__(self, data:np.ndarray, labels:np.ndarray, k:float, C:float, c:float, d:float):
    super().__init__(data, labels, k, C)
    self.c = c
    self.d = d

  def H(self) -> np.ndarray:
    kernel = polynomial_kernel(self.data, self.data, self.k, self.c, self.d)
    z = np.dot(v_col(self.z), v_row(self.z))
    return z * kernel
  
  def scores(self, X:np.ndarray):
    if(self.is_fitted is False):
      self.train()
    return (v_col(self.alpha) * v_col(self.z) * polynomial_kernel(self.data, X, self.k, self.c, self.d)).sum(0)
  
class test():
  def f1(self, x:np.ndarray) -> float:
    return (x[0]+3)**2+np.sin(x[0])+(x[1]+1)**2
  
  def f2(self, x:np.ndarray) -> np.ndarray:
    return (x[0]+3)**2+np.sin(x[0])+(x[1]+1)**2, v_col(np.array([2*(x[0]+3)+np.cos(x[0]), 2*(x[1]+1)]))

  def approx_optimization(self):
    return fmin_l_bfgs_b(self.f1, v_col(np.array([0,0])), approx_grad=True, iprint=True)
    
  def optimization(self):
    return fmin_l_bfgs_b(self.f2, v_col(np.array([0,0])), iprint=True)
