from utils import *
from plots import *
from preprocessing import *
import numpy as np
from preprocessing import *
from models.gaussian_models import *

# def load_csv(file_name: str) -> tuple:
#   data = []
#   labels = []
#   with open(file_name, 'r') as opened_file:
#     for line in opened_file:
#       sample = [float(i) for i in line.rstrip().split(',')[0:-1]]
#       data.append(sample)
#       labels.append(line.rstrip().split(',')[-1])
#   keys = np.unique(labels)
#   labels = np.array(labels)
#   l = np.zeros(labels.size)
#   for i in range(keys.size):
#     l[labels==keys[i]] = i
#   return np.array(data).T, l

if __name__=="__main__":
  dtr,ltr = load_csv('data/Train.txt')

  #TEST Nico
  print("DTR - shape: ", dtr.shape)
  print("LTR - shape: ", ltr.shape)

  print("#############################################")

  print("mean - shape: ", mean(dtr).shape)
  print("variance - shape: ", variance(dtr).shape)
  print("covariance_matrix - shape: ", covariance_matrix(dtr).shape)
  print("classes_number: ", classes_number(ltr))

  print("#############################################")

  # print("mean_of_classes - length: (", str(len(compute_mean_of_classes(dtr, ltr)[0])) + ", " + str(len(compute_mean_of_classes(dtr, ltr)[1])) + " )")
  # print("covariance_of_classes - shape: (", str(compute_covariance_of_classes(dtr, ltr)[0].shape) + ", " + str(compute_covariance_of_classes(dtr, ltr)[1].shape) + " )")

  # print("mean_of_classes: " + str(compute_mean_of_classes(dtr, ltr)))
  # print("covariance_of_classes: " + str(compute_covariance_of_classes(dtr, ltr)))

  print("#############################################")

  print("Computing PCA...")
  PCA = pca(dtr)
  PCA.process()

  print("Computing LDA...")
  LDA = lda(dtr, ltr)
  LDA.process()

  print("#############################################")

  print("Computing MVG...")
  MVG = mvg(dtr, ltr)
  MVG.fit()

  print("Computing NBG...")
  NBG = naiveg(dtr, ltr)
  NBG.fit()

  print("Computing TIEDG...")
  TIEDG = tiedg(dtr, ltr)
  TIEDG.fit()

  # mean_f1 = dtr[0, :].mean()
  # print("mean_f1: ", mean_f1)

  # var = 0
  # for feature in dtr[0, :]:
  #   var += (feature - mean_f1)**2

  # var /= dtr[0, :].size

  # print("var_f1: ", var)

  # print(dtr[:, 0].shape)
  # print(dtr[:, 0].sum() / dtr[:, 0].size)
  # print(dtr[:, 0].size)

  # print("#############################################")

  # print(v_col(dtr.mean(1)).shape)

  # print("#############################################")

  # print(dtr[:, 0])

  # create pac and lda objects
  # PCA = pca(dtr)
  # LDA = lda(dtr, ltr)

  # process the data
  # PCA.process()
  # LDA.process()

  # compute plots and histograms
  # generate_and_save_features_hists(dtr, ltr)
  # generate_and_save_features_scatter_plots(dtr, ltr)
  # pca_explained_variance(PCA)
  # heatmaps(dtr, ltr)
  # lda_direction_histogram(LDA)

  #dtr,ltr = load_csv('data/Train.txt')
  
  """sol_pca = np.load('IRIS_PCA_matrix_m4.npy')
  sol_lda = np.load('IRIS_LDA_matrix_m2.npy')
  pca = pca(dtr)
  pca.process(4)
  lda = lda(dtr, ltr)
  lda.process()
  print(f'PCA error: {(sol_pca-pca.eigenvectors[:,:4]).max()}')
  print(f'LDA error: {(sol_lda-lda.eigenvectors[:,:2]).max()}') """
  

  
