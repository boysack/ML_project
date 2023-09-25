
import numpy as np

def load_csv(file_name: str) -> tuple:
  data = []
  labels = []
  with open(file_name, 'r') as opened_file:
    for line in opened_file:
      sample = [float(i) for i in line.rstrip().split(',')[0:-1]]
      label = [float(i) for i in line.rstrip().split(',')[-1]]
      data.append(sample)
      labels.append(label)
  return np.array(data).T, np.array(labels)