
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

def z_normalization(samples, mean=None, var=None) -> tuple:
  if(not mean):
    mean = samples.mean(axis=1).reshape(samples.shape[0],1)
  if(not var):
    var = samples.var(axis=1).reshape(samples.shape[0],1)
  return (samples-mean)/var

  