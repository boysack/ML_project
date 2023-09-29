from utils import *
from plots import *
from preprocessing import *
import numpy as np
from preprocessing import *

if __name__=="__main__":
  dtr,ltr = load_csv('data/Train.txt')
  heatmaps(dtr, ltr)