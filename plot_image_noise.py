import matplotlib.pyplot as plt
import numpy as np


def show_img_noise(image, noise, predictImage=-1, predictNoise=-1, predictAdded=-1):
	"""
	plots the image, the noise and the added image + noise besides eachother to
	compare those
	"""
	f, axarr = plt.subplots(1,3, figsize=(15, 4), sharex=True)
	axarr[0].imshow(image[:,:,0], vmin=0, vmax=1)
	noiseScale = 1./np.max(noise[:,:,0])
	axarr[1].imshow(noise[:,:,0] * noiseScale, vmin=0, vmax=1)
	imgPNoise = image[:,:,0]+noise[:,:,0]
	scaleIN = 1./np.max(imgPNoise)
	axarr[2].imshow(imgPNoise * scaleIN, vmin=0, vmax=1)
	if (predictImage != -1):
		axarr[0].set_title('Prediction Image: ' + str(predictImage))
	else:
		axarr[0].set_title('Image:')
	if (predictNoise != -1):
		axarr[1].set_title('Prediction Noise: ' + str(predictNoise) + "; scaling: " + str(noiseScale))
	else:
		axarr[1].set_title('Noise; scaling: ' + str(noiseScale))
	if (predictAdded != -1):
		axarr[2].set_title('Prediction Image + Noise: ' + str(predictAdded) + "; scaling = " + str(scaleIN))
	else:
		axarr[2].set_title('Image + Noise; scaling = ' + str(scaleIN))
	plt.show()