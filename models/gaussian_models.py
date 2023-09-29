from abc import abstractmethod
import numpy as np

class gaussian:
  def __init__(self, data:np.ndarray, labels:np.ndarray):
    self.data = data
    self.labels = labels

  @abstractmethod
  def fit():
    pass

  @abstractmethod
  def predict():
    pass

class mvg(gaussian):
  def __init__(self, data:np.ndarray, labels:np.ndarray):
    super().__init__(data, labels)

class naiveg(gaussian):
  def __init__(self, data:np.ndarray, labels:np.ndarray):
    super().__init__(data, labels)

class tiedg(gaussian):
  def __init__(self, data:np.ndarray, labels:np.ndarray):
    super().__init__(data, labels)