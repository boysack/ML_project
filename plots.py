import matplotlib.pyplot as plt
import os
import numpy as np
import scipy as sp
from utils import *
from preprocessing import *

# Definition of a dictionary to map values to color maps
heatmap_colors = {
  -1: 'gray_r',
  0: 'Reds',
  1: 'Blues'
}

def generate_and_save_features_hists(samples, labels, pathname='./'):
  r=(samples.min(), samples.max())
  b=round((samples.max()-samples.min())/0.04)
  for i in range(samples.shape[0]):
    plt.hist(samples[i, labels==0], range=r, bins=b, alpha=0.5, color='red', label='male (0)', density=True)
    plt.hist(samples[i, labels==1], range=r, bins=b, alpha=0.5, color='blue', label='female (1)', density=True)
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

def pca_explained_variance(pca:pca) -> None:
  if pca.is_preprocessed is False:
    pca.process()

  pca_explained_variance_ratio = np.cumsum(pca.eigenvalues) / np.sum(pca.eigenvalues)
  plt.plot(np.arange(1, pca.data.shape[0] + 1), pca_explained_variance_ratio, 'o-')
  plt.xlabel('PCA dimensions')
  plt.ylabel('Fraction of explained variance')
  plt.grid()
  plt.title('PCA explained variance')
  plt.savefig('./plots/pca_explained_variance/pca_explained_variance')
  plt.clf()


def heatmap_creation(data: np.ndarray, color: int = -1) -> None:
  heatmap = np.zeros((data.shape[0], data.shape[0]))
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      if j <= i:
        heatmap[i][j] = abs(sp.stats.pearsonr(data[i, :], data[j, :])[0])
        heatmap[j][i] = heatmap[i][j]

  heatmap_plot = plt.imshow(heatmap, cmap=heatmap_colors[color])
  plt.colorbar(heatmap_plot)
  plt.xticks(np.arange(0, data.shape[0]), np.arange(1, data.shape[0] + 1))
  plt.yticks(np.arange(0, data.shape[0]), np.arange(1, data.shape[0] + 1))
  plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
  
  # plt.show()
  if color == -1:
    plt.title('Heatmap for the whole dataset', pad=20)
    plt.savefig('./plots/heatmaps/heatmap')
  else:
    # male = 0 female = 1
    plt.title('Heatmap for class ' + str(color), pad=20)
    plt.savefig('./plots/heatmaps/heatmap_class_' + str(color))
  plt.clf()

def heatmaps(data: np.ndarray, labels: np.ndarray) -> None:
  for i in range(classes_number(labels)):
    heatmap_creation(data[:, labels == i], i)

  heatmap_creation(data)

def lda_direction_histogram(lda: lda) -> None:
  if lda.is_preprocessed is False:
     lda.process()
  for i in range(lda.data.shape[0]):
    plt.hist(lda.data[i, lda.labels==0], bins=50, color='red', label='male (0)', density=True, alpha=0.5)
    plt.hist(lda.data[i, lda.labels==1], bins=50, color='blue', label='female (1)', density=True, alpha=0.5)
    plt.legend(loc='upper right')
    plt.savefig('./plots/lda/lda_plot')
    plt.clf()
