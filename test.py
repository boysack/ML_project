from models.gaussian_models import *
from models.support_vector_machine import *
from models.logistic_regression import *

import numpy as np

def load_csv(file_name: str) -> tuple:
  data = []
  labels = []
  with open(file_name, 'r') as opened_file:
    for line in opened_file:
      sample = [float(i) for i in line.rstrip().split(',')[0:-1]]
      data.append(sample)
      labels.append(line.rstrip().split(',')[-1])
  keys = np.unique(labels)
  labels = np.array(labels)
  l = np.zeros(labels.size)
  for i in range(keys.size):
    l[labels==keys[i]] = i
  return np.array(data).T, l

def load_iris_binary():
  D, L = load_csv("iris.csv")
  D = D[:, L != 0] # We remove setosa from D
  L = L[L!=0] # We remove setosa from L
  L[L==2] = 0 # We assign label 0 to virginica (was label 2) return D, L
  return D,L

def split_db_2to1(D, L, seed=0):
  nTrain = int(D.shape[1]*2.0/3.0)
  np.random.seed(seed)
  idx = np.random.permutation(D.shape[1]) 
  idxTrain = idx[0:nTrain]
  idxTest = idx[nTrain:]

  DTR = D[:, idxTrain]
  DTE = D[:, idxTest]
  LTR = L[idxTrain]
  LTE = L[idxTest]

  return (DTR, LTR), (DTE, LTE)

def logistic_test():
  l = [10**-6, 10**-3, 10**-1, 1]
  for i in range(len(l)):
    print(f"for lambda value -> {l[i]}")
    lr = logistic_regression(dtr, ltr, l[i])
    scores = lr.scores(dte)
    lpr = []
    for i in range(lte.size):
      if(scores[i]>=0):
        lpr.append(1)
      else:
        lpr.append(0)
    print(1-((lpr==lte).sum()/lte.size))

if __name__=="__main__":

  #dtr, ltr = load_csv("iris.csv")
  
  """ print("############")
  print("## approx ##")
  print("############")
  t = test()
  x,f,d = t.approx_optimization()
  print(x)
  print(d)

  print("############")
  print("# !approx! #")
  print("############")
  
  x,f,d = t.optimization()
  print(x)
  print(d)"""
  
  D, L = load_iris_binary()
  (dtr,ltr),(dte,lte) = split_db_2to1(D,L)
  """ svm = soft_svm(dtr,ltr,1,.1)
  svm.train() """
  """ print(svm.weights)
  print(svm.bias) """

  K = [1,1,1,10,10,10]
  C = [.1,1.0,10.0,.1,1.0,10.0]
  for i in range(6):
    svm_1 = hard_svm(dtr, ltr, K[i], C[i])
    spr_1 = svm_1.scores(dte)
    lpr_1 = (spr_1 >= 0).astype(int)
    print(f'K: {K[i]:02} | C: {C[i]:04} | ERR: {(1-(lpr_1 == lte).sum()/lte.size)*100:.1f}%')
  
  params = [(0.0,1.0,2.0,0.0),(1.0,1.0,2.0,0.0),(0.0,1.0,2.0,1.0),(1.0,1.0,2.0,1.0)]

  print("Polynomial SVM test")
  for i in range(4):
    svm_2 = polynomial_svm(dtr, ltr, params[i][0], params[i][1], params[i][3], params[i][2])
    spr_2 = svm_2.scores(dte)
    lpr_2 = (spr_2 >= 0).astype(int)
    print(f'K: {params[i][0]:02} | C: {params[i][1]:04} | d: {params[i][2]:04} | c: {params[i][3]:04} | ERR: {(1-(lpr_2 == lte).sum()/lte.size)*100:.1f}%')


  params = [(0.0,1.0,1.0),(0.0,1.0,10.0),(1.0,1.0,1.0),(1.0,1.0,10.0)]

  print("Radial Basis Function SVM test")
  for i in range(4):
    svm_3 = radial_basis_function_svm(dtr, ltr, params[i][0], params[i][1], params[i][2])
    spr_3 = svm_3.scores(dte)
    lpr_3 = (spr_3 >= 0).astype(int)
    print(f'K: {params[i][0]:02} | C: {params[i][1]:04} | gamma: {params[i][2]:04} | ERR: {(1-(lpr_3 == lte).sum()/lte.size)*100:.1f}%')
  