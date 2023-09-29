from utils import *
from plots import *
from preprocessing import *
import numpy as np
from preprocessing import *

if __name__=="__main__":
  dtr,ltr = load_csv('data/Train.txt')
  # generate_and_save_features_hists(z_normalization(dtr), ltr, './plots/histograms/')
  # generate_and_save_features_scatter_plots(z_normalization(dtr), ltr, './plots/scatter_plots/')
  # data, labels = k_fold(dtr, ltr)
  # print(dtr)
  # print(data)
  # print(ltr)
  # print(labels)
  
  # creates three heatmaps (class 0, class 1, and both classes)
  # heatmaps(dtr, ltr)

  # plots the Histogram of dataset features - LDA direction
  # lda_direction_histogram(lda(dtr, ltr))
  # generate_and_save_features_hists(LDA(dtr, ltr), ltr, './plots/histograms/lda/')

  # t = compute_mean_of_classes(dtr,ltr)

  # pca_explained_variance(pca(dtr))