from utils import *
from plots import *
import numpy as np

if __name__=="__main__":
  dtr,ltr = load_csv('data/Train.txt')
  # generate_and_save_features_hists(z_normalization(dtr), ltr, './plots/histograms/')
  # generate_and_save_features_scatter_plots(z_normalization(dtr), ltr, './plots/scatter_plots/')
  heatmap_creation(dtr)

  # print(compute_mean_of_classes(dtr, ltr))


