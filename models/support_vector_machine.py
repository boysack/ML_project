from abc import abstractmethod
import numpy as np
from utils import *

from scipy.optimize import fmin_l_bfgs_b

class svm:
  def __init__(self, data:np.ndarray, labels:np.ndarray, k:float, C:float) -> None:
        # capire se salvare direttamente col k inserito o inserirlo occasionalmente
        self.data = data
        self.labels = labels
        self.z = 2*labels-1
        self.k = k
        self.C = C

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

class soft_svm(svm):
  def __init__(self, data:np.ndarray, labels:np.ndarray, k:float, C:float):
    super().__init__(data, labels, k, C)

  # forse in superclasse?
  def h_hat(self) -> np.ndarray:
    d = np.vstack((self.data, [self.k] * self.data.shape[1]))
    d = d * self.z
    h_hat = np.dot(d.T, d)
    return h_hat
  
  # forse in superclasse?
  def alpha_to_w_b(self, alpha:np.ndarray) -> np.ndarray:
    d = np.vstack((self.data, [self.k] * self.data.shape[1]))
    return np.sum((v_col(alpha) * v_col(self.z)).T * d, axis=1)

  # wrapper?
  def obj(self, alpha:np.ndarray) -> np.ndarray:
    h_hat = self.h_hat()
    obj_f = .5 * np.dot(np.dot(alpha.T, h_hat), alpha) - np.dot(alpha.T, np.diag(np.ones((self.data.shape[1],self.data.shape[1]))))
    gradient_f = v_col(np.dot(h_hat, alpha))-v_col(np.ones(self.data.shape[1]))
    return (obj_f, gradient_f)
  
  def train(self):
    alpha = np.zeros((self.data.shape[1], 1))
    box_const = (0, self.C)
    alpha, _, _ = fmin_l_bfgs_b(self.obj, alpha, bounds=[box_const]*self.data.shape[1])
    self.alpha = alpha
    w_hat = self.alpha_to_w_b(alpha)
    self.weights = w_hat[:-1]
    self.bias = w_hat[-1]
    
    self.is_fitted = True

  def scores(self, X:np.ndarray):
    if(self.is_fitted is False):
      self.train()
    return np.dot(self.weights.T, X) + self.bias

class hard_svm(svm):
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
