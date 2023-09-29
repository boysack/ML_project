import matplotlib.pyplot as plt
import os
import numpy as np
import scipy as sp
from utils import *
from preprocessing import *

def generate_and_save_features_hists(samples, labels, pathname='./'):
  r=(samples.min(), samples.max())
  b=round((samples.max()-samples.min())/0.04)
  for i in range(samples.shape[0]):
    plt.hist(samples[i, labels==0], range=r, bins=b, alpha=0.5, color='red', label='F', density=True)
    plt.hist(samples[i, labels==1], range=r, bins=b, alpha=0.5, color='blue', label='T', density=True)
    plt.legend(loc='upper right')
    if(i<10):
      index = '0' + str(i)
    else:
      index = str(i)
    # plt.savefig(pathname + 'feature' + index + '_hist')
    plt.clf()
    plt.show()

def generate_and_save_features_scatter_plots(samples, labels, pathname='./'):
  for i in range(samples.shape[0]):
    for j in range(samples.shape[0]):
      if i==j:
        continue
      plt.xlim(samples.min()-0.2, samples.max()+0.2)
      plt.ylim(samples.min()-0.2, samples.max()+0.2)
      plt.scatter(samples[i, labels==0], samples[j, labels==0], alpha=0.5, color='red', label='F')
      plt.scatter(samples[i, labels==1], samples[j, labels==1], alpha=0.5, color='blue', label='T')
      if(i<10):
        index_i = '0' + str(i)
      else:
        index_i = str(i)
      if(j<10):
        index_j = '0' + str(j)
      else:
        index_j = str(j)
      plt.savefig(pathname + 'features' + index_i + 'x' + index_j + '_scatter')
      plt.clf()

def pca_explained_variance(data, labels):
  pass

def heatmap_creation(data: np.ndarray) -> None:
  heatmap = np.zeros((data.shape[0], data.shape[0]))
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      if j <= i:
        heatmap[i][j] = abs(sp.stats.pearsonr(data[i, :], data[j, :])[0])
        heatmap[j][i] = heatmap[i][j]

  heatmap_plot = plt.imshow(heatmap, cmap='gray_r')
  plt.colorbar(heatmap_plot)
  plt.xticks(np.arange(0, data.shape[0]), np.arange(1, data.shape[0] + 1))
  plt.yticks(np.arange(0, data.shape[0]), np.arange(1, data.shape[0] + 1))
  plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
  # plt.show()
  plt.title('Pearson correlation coefficient for the dataset features', pad=20)
  plt.show()

def heatmaps(data: np.ndarray, labels: np.ndarray) -> None:
  for i in range(classes_number(labels)):
    heatmap_creation(data[:, labels == i])

  heatmap_creation(data)

def lda_direction_histogram(data: np.ndarray, labels: np.ndarray) -> None:
  projected_data = LDA(data, labels)
  for i in range(data.shape[0]):
    plt.hist(projected_data[i, labels==0], bins=200, color='red', label='F', density=True)
    plt.hist(projected_data[i, labels==1], bins=200, color='blue', label='T', density=True)
    plt.show()