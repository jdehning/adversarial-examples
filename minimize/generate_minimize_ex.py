import keras
import tensorflow as tf
from keras.models import load_model
from keras import losses
import pandas as pd
import numpy as np
from tqdm import tqdm #tqdm is a library to display a progress bar
import matplotlib.pyplot as plt
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from plot_image_noise import show_img_noise
import scipy.optimize, scipy.stats
import time

#instanciate the session at the loading of this module. Could perhaps lead to problems later? But instanciating the
#session in a function doesn't work for unknown reasons.
sess = tf.Session()
from keras import backend as K
K.set_session(sess)


def read_data_mnist():
	"""
	read the MNIST traning set\n
	\n
	returns: (dataX, dataY)\n
	dataX: array of (28, 28, 1) arrays which are the image pixels. Each
	pixel has an intensity in range [0,1]\n
	dataY: classification: [1, ..., 9]\n
	"""
	train = pd.read_csv("../data/train.csv").values
	data_X = train[:, 1:].reshape(train.shape[0], 28, 28, 1)
	data_X = data_X.astype(float)
	data_X /= 255.0
	data_Y = keras.utils.to_categorical(train[:, 0])
	return data_X, data_Y

def minimize_func (r, c, image, target, loss_func):
	"""
	the function that is going to be minimized\n
	\n
	r: (x, y, colors) array with the pertubation values\n
	c: float factor that gets minimized through an additional line search\n
	image: (x, y, colors) array with the image pixel values for all available colors\n
	target: the target classification\n
	loss_func: used loss function of the neuronal network. The function should expect\n
	a pixel array (x, y, color) and a target and return an error. Needs to be a
	continious function.\n
	returns: value to be minimized
	"""
	#print("d")
	return vec_abs(image + r)**2
	#return c * vec_abs(r) + loss_func(image + r, target)[0]

def vec_abs(arr):
	"""
	returns the absolute value of the array which is interpreted as a vector
	"""
	return np.sqrt(np.sum(arr*arr))

def create_loss_func_for_minimize(model):
	"""
	creates a loss function from the keras model to be passed on to the minimize_func\n
	returns: function to be passed on to the minimize_func. This function expects an
		image and the target prediction and returns an numpy array containing the
		error.
	"""
	loss = model.loss
	if(type(loss) is str):
		loss = getattr(losses, loss)
	def loss_func(image, target):
		#print("e")
		prediction = model.predict(np.array([image], dtype="float32"), batch_size=1, verbose=0)
		prediction = tf.convert_to_tensor(prediction)
		target = tf.convert_to_tensor(np.array([target], dtype="float32"))
		#print("f")
		return loss(prediction, target).eval(session=sess)
	return loss_func

def run_batch_minimize(model, images, truePrediction):
	loss_func = create_loss_func_for_minimize(model)
	i = 0
	c = 20.
	rs = []
	imgShape = np.shape(images[0])
	def cb(a):
		print ("a")
	def to_minimize(r):
		if (np.shape(r) != imgShape):
			r = np.reshape(r, imgShape)
		#print("c")
		temp = minimize_func(r, c, image, np.append(truePrediction[i][4:],truePrediction[i][:4]), loss_func)
		print(temp)
		return temp

	def constrain_func(r):
		return loss_func()
	constrains = {
		fun: loss_func
	}
	for image in images:
		#print (to_minimize(np.array(np.random.rand(*imgShape), dtype="float32")))
		rs.append( scipy.optimize.minimize(to_minimize, x0=np.random.rand(*imgShape)/10., 
			method="L-BFGS-B", callback=cb), tol=0.1, options={maxiter: 2} )
		i += 1
	return np.array(rs)

model = load_model('../keras_model1')
dataX, dataY = read_data_mnist()
print (create_loss_func_for_minimize(model)(np.array(np.random.rand(28,28,1), dtype="float32"), dataY[1]))
rs = run_batch_minimize(model, np.array([dataX[1]]), np.array([dataY[1]]))
show_img_noise(dataX[1], rs[0])