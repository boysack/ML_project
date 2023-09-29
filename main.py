from utils import *
from plots import *
import numpy as np
from preprocessing import *

if __name__=="__main__":
  dtr,ltr = load_csv('data/Train.txt')
  # generate_and_save_features_hists(z_normalization(dtr), ltr, './plots/histograms/')
  # generate_and_save_features_scatter_plots(z_normalization(dtr), ltr, './plots/scatter_plots/')
  
  # creates three heatmaps (class 0, class 1, and both classes)
  # heatmaps(dtr, ltr)

  # plots the Histogram of dataset features - LDA direction
  lda_direction_histogram(dtr, ltr)
  # generate_and_save_features_hists(LDA(dtr, ltr), ltr, './plots/histograms/lda/')

