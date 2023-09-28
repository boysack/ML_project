
import numpy as np

def load_csv(file_name: str) -> tuple:
  data = []
  labels = []
  with open(file_name, 'r') as opened_file:
    for line in opened_file:
      sample = [float(i) for i in line.rstrip().split(',')[0:-1]]
      data.append(sample)
      labels.append(int(line.rstrip().split(',')[-1]))
  return np.array(data).T, np.array(labels)

def z_normalization(x, mean=None, var=None) -> np.ndarray:
  centered_data = center_data(x, mean)
  if(not var):
    var = variance(x)
  return (center_data)/var

def v_col(x: np.ndarray) -> np.ndarray:
  return x.reshape((x.size, 1))

def v_row(x: np.ndarray) -> np.ndarray:
  return x.reshape((1, x.size))

def mean(data: np.ndarray) -> np.ndarray:
  return v_col(data.mean(1))

def variance(data: np.ndarray) -> np.ndarray:
  return v_col(data.var(1))

def center_data(data: np.ndarray, mean:np.ndarray=None) -> np.ndarray:
  if(mean==None):
    mean=mean(data)
  return data - mean

def covariance_matrix(data: np.ndarray) -> np.ndarray:
  centered_data = center_data(data)
  return (1 / data.shape[1]) * np.dot(centered_data, centered_data.T)

def classes_number(labels: np.ndarray):
  return np.unique(labels).size

def compute_mean_of_classes(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
  classes = classes_number(labels)
  mean_of_classes = np.zeros((data.shape[0], classes))
  for i in range(classes):
      mean_of_classes[:, i] = mean(data[:, labels == i])
  return mean_of_classes