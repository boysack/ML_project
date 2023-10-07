from abc import abstractmethod
import numpy as np
from utils import *

from scipy.optimize import fmin_l_bfgs_b

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
  def obj():
    pass

  @abstractmethod
  def train():
    pass

  @abstractmethod
  def scores():
    pass

class hard_svm(svm):
  def __init__(self, data:np.ndarray, labels:np.ndarray, k:float, C:float):
    super().__init__(data, labels, k, C)

  # forse in superclasse?
  # forse si può reimplementare solo questa parte per implementare i kernel
  def H_hat(self) -> np.ndarray:
    X_hat = np.vstack((self.data, [self.k] * self.data.shape[1]))
    X_hat = X_hat * self.z
    return np.dot(X_hat.T, X_hat)
  
  # forse in superclasse?
  def alpha_to_w_b(self, alpha:np.ndarray) -> np.ndarray:
    X_hat = np.vstack((self.data, [self.k] * self.data.shape[1]))
    return np.sum((v_col(alpha) * v_col(self.z)).T * X_hat, axis=1)
    
  # wrapper?
  def obj(self, alpha:np.ndarray) -> np.ndarray:
    H_hat = self.H_hat()
    obj_f = .5 * np.dot(np.dot(alpha.T, H_hat), alpha) - np.dot(alpha.T, v_col(np.ones((self.data.shape[1]))))
    gradient_f = v_col(np.dot(H_hat, alpha))-v_col(np.ones(self.data.shape[1]))
    return obj_f, gradient_f
  
  def train(self):
    alpha = np.zeros((self.data.shape[1], 1))
    box_const = (0, self.C)
    alpha, _, _ = fmin_l_bfgs_b(self.obj, alpha, factr=1.0, bounds=[box_const]*self.data.shape[1])
    self.alpha = alpha
    w_hat = self.alpha_to_w_b(alpha)
    self.weights = w_hat[:-1]
    self.bias = w_hat[-1]

    self.is_fitted = True

  def scores(self, X:np.ndarray):
    if(self.is_fitted is False):
      self.train()
    return np.dot(self.weights.T, X) + self.bias * self.k

class soft_svm(svm):
  def __init__(self, data:np.ndarray, labels:np.ndarray, ):
    pass

class test():
   
  def f1(self, x:np.ndarray) -> float:
    return (x[0]+3)**2+np.sin(x[0])+(x[1]+1)**2
  
  def f2(self, x:np.ndarray) -> np.ndarray:
    return (x[0]+3)**2+np.sin(x[0])+(x[1]+1)**2, v_col(np.array([2*(x[0]+3)+np.cos(x[0]), 2*(x[1]+1)]))

  def approx_optimization(self):
    return fmin_l_bfgs_b(self.f1, v_col(np.array([0,0])), approx_grad=True, iprint=True)
    
  def optimization(self):
    return fmin_l_bfgs_b(self.f2, v_col(np.array([0,0])), iprint=True)
