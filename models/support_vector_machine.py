from abc import abstractmethod
import numpy as np
from utils import *

from scipy.optimize import fmin_l_bfgs_b

class discriminative_model:
  def __init__(self, data:np.ndarray, labels:np.ndarray, l:int) -> None:
        self.data = data
        self.labels = labels
        self.l = l
        self.weights = None
        self.bias = None
        self.score_values = None
        self.is_fitted = False

  @abstractmethod
  def train():
    pass

  @abstractmethod
  def scores():
    pass

  @abstractmethod
  def objective_function():
    pass

class soft_svm(discriminative_model):
  def __init__(self, data:np.ndarray, labels:np.ndarray, ):
    pass

class hard_svm(discriminative_model):
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
