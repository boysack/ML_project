from utils import *
from plots import *
from preprocessing import *
import numpy as np
from preprocessing import *

def load_csv(file_name: str) -> tuple:
  data = []
  labels = []
  with open(file_name, 'r') as opened_file:
    for line in opened_file:
      sample = [float(i) for i in line.rstrip().split(',')[0:-1]]
      data.append(sample)
      labels.append(line.rstrip().split(',')[-1])
  keys = np.unique(labels)
  labels = np.array(labels)
  l = np.zeros(labels.size)
  for i in range(keys.size):
    l[labels==keys[i]] = i
  return np.array(data).T, l

if __name__=="__main__":
  dtr,ltr = load_csv('data/Train.txt')
  PCA = pca(dtr)
  LDA = lda(dtr, ltr)

  PCA.process()
  LDA.process()

  # print(PCA.eigenvectors)
  # print("Covariance matrix using numpy: ")
  # print(np.cov(dtr))
  # print(LDA.eigenvectors)
  # generate_and_save_features_hists(dtr, ltr)
  # generate_and_save_features_scatter_plots(dtr, ltr)
  # pca_explained_variance(PCA)
  # heatmaps(dtr, ltr)
  lda_direction_histogram(LDA)

  #dtr,ltr = load_csv('data/Train.txt')
  dtr, ltr = load_csv('iris.csv')
  
  """sol_pca = np.load('IRIS_PCA_matrix_m4.npy')
  sol_lda = np.load('IRIS_LDA_matrix_m2.npy')
  pca = pca(dtr)
  pca.process(4)
  lda = lda(dtr, ltr)
  lda.process()
  print(f'PCA error: {(sol_pca-pca.eigenvectors[:,:4]).max()}')
  print(f'LDA error: {(sol_lda-lda.eigenvectors[:,:2]).max()}') """
  

  
