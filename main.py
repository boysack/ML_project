from utils import *
from plots import *
from preprocessing import *
import numpy as np
from preprocessing import *

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

