from utils import *
from plots import *
from preprocessing import *
import numpy as np

if __name__=="__main__":
  dtr,ltr = load_csv('data/Train.txt')
  center_data(dtr)
  # generate_and_save_features_hists(z_normalization(dtr), ltr, './plots/histograms/')
  # generate_and_save_features_scatter_plots(z_normalization(dtr), ltr, './plots/scatter_plots/')
  """   pca = pca(dtr, dtr.shape[0])
  pca.process()
  print(pca.data)

  lda = lda(dtr, ltr, np.unique(ltr).size-1)
  lda.process()
  print(lda.data) """

  t = compute_mean_of_classes(dtr,ltr)
  print(t)