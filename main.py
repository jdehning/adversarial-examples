import keras
import numpy as np


trainData = np.loadtxt("./data/train.csv", skiprows=1, delimiter=",")
#Each line contains one image, 28x28=784 px. first Column is the number displayed.
#	print(np.shape(trainData))
#	(42000, 785)

