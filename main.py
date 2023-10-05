from utils import *
from plots import *
from preprocessing import *
import numpy as np
from preprocessing import *
from models.gaussian_models import *
from models.logistic_regression import *
import time

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

  start_time_load = time.time()
  dtr,ltr = load_csv('data/Train.txt')
  end_time_load = time.time()

  #TEST Nico
  print("DTR - shape: ", dtr.shape)
  print("LTR - shape: ", ltr.shape)

  print("#############################################")

  print("mean - shape: ", mean(dtr).shape)
  print("variance - shape: ", variance(dtr).shape)
  print("covariance_matrix - shape: ", covariance_matrix(dtr).shape)
  print("classes_number: ", classes_number(ltr))

  print("#############################################")

  print("mean_of_classes - length: (", str(len(compute_mean_of_classes(dtr, ltr)[0])) + ", " + str(len(compute_mean_of_classes(dtr, ltr)[1])) + " )")
  print("covariance_of_classes - shape: (", str(compute_covariance_of_classes(dtr, ltr)[0].shape) + ", " + str(compute_covariance_of_classes(dtr, ltr)[1].shape) + " )")

  # print("mean_of_classes: " + str(compute_mean_of_classes(dtr, ltr)))
  # print("covariance_of_classes: " + str(compute_covariance_of_classes(dtr, ltr)))

  print("#############################################")

  print("Computing PCA...")
  PCA = pca(dtr)

  start_time_pca = time.time()
  PCA.process()
  end_time_pca = time.time()

  print("Computing LDA...")
  LDA = lda(dtr, ltr)

  start_time_lda = time.time()
  LDA.process()
  end_time_lda = time.time()

  print("#############################################")

  print("Computing MVG...")
  MVG = mvg(dtr, ltr)

  start_time_mvg = time.time()
  MVG.train()
  end_time_mvg = time.time()

  result_MVG = MVG.scores(dtr)
  print("result_MVG: ", result_MVG)

  print("#############################################")

  print("Computing NBG...")
  NBG = naiveg(dtr, ltr)

  start_time_nbg = time.time()
  NBG.train()
  end_time_nbg = time.time()

  result_NBG = NBG.scores(dtr)
  print("result_NBG: ", result_NBG)

  print("#############################################")

  print("Computing TIEDG...")
  TIEDG = tiedg(dtr, ltr)

  start_time_tiedg = time.time()
  TIEDG.train()
  end_time_tiedg = time.time()

  result_TIEDG = TIEDG.scores(dtr)
  print("result_TIEDG: ", result_TIEDG)

  print("#############################################")

  print("Computing LR...")
  LR = logistic_regression(dtr, ltr, 0.1)

  start_time_lr = time.time()
  LR.scores()
  end_time_lr = time.time()

  result_LR = LR.score_values
  print("result_LR: ", result_LR)

  results_lr = []

  for value in result_LR:
    if value > 0:
      results_lr.append(1)
    else:
      results_lr.append(0)

  correct = 0

  for result, label in zip(results_lr, ltr):
    if result == label:
      correct += 1

  print("number of correct predictions: ", correct, " out of ", len(results_lr), " samples")
  print("accuracy: ", correct / len(results_lr))

  print("#############################################")

  print("Computing QLR...")
  QLR = logistic_regression(dtr, ltr, 0.1, True)
  
  start_time_qlr = time.time()
  QLR.scores()
  end_time_qlr = time.time()

  result_QLR = QLR.score_values
  print("result_QLR: ", result_QLR)

  results_qlr = []

  for value in result_QLR:
    if value > 0:
      results_qlr.append(1)
    else:
      results_qlr.append(0)

  correct_qlr = 0

  for result, label in zip(results_qlr, ltr):
    if result == label:
      correct_qlr += 1

  print("number of correct predictions: ", correct_qlr, " out of ", len(results_qlr), " samples")
  print("accuracy: ", correct_qlr / len(results_qlr))

  print("#############################################")

  print("Execution times:")
  print("load: ", end_time_load - start_time_load)
  print("pca: ", end_time_pca - start_time_pca)
  print("lda: ", end_time_lda - start_time_lda)
  print("mvg: ", end_time_mvg - start_time_mvg)
  print("nbg: ", end_time_nbg - start_time_nbg)
  print("tiedg: ", end_time_tiedg - start_time_tiedg)
  print("lr: ", end_time_lr - start_time_lr)
  print("qlr: ", end_time_qlr - start_time_qlr)

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
  

  
