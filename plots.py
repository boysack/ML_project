import matplotlib.pyplot as plt
import os

def generate_and_save_features_hists(samples, labels, pathname='./'):
  r=[samples.min(), samples.max()]
  b=round((samples.max()-samples.min())/0.04)
  for i in range(samples.shape[0]):
    plt.hist(samples[i, labels==0], range=r, bins=b, alpha=0.5, color='red', label='F', density=True)
    plt.hist(samples[i, labels==1], range=r, bins=b, alpha=0.5, color='blue', label='T', density=True)
    plt.legend(loc='upper right')
    if(i<10):
      index = '0' + str(i)
    else:
      index = str(i)
    plt.savefig(pathname + 'feature' + index + '_hist')
    plt.clf()

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
