from utils import *
from plots import *
from preprocessing import *
import numpy as np
from preprocessing import *

if __name__=="__main__":
  dtr,ltr = load_csv('data/Train.txt')
  # prova = lda(dtr, ltr)
  # prova.process(2)
  # prova.process(1)
  # prova.process(0)
  heatmaps(dtr, ltr)
  lda_direction_histogram(lda(dtr, ltr))