import matplotlib.pyplot as plt
import numpy as np
import constants

def show_img_noise(image, noise, predictImage=-1, predictNoise=-1, predictAdded=-1, color = False, save_as = False, std_noise=-1,
                   prob_image = None, prob_adv_ex = None):
    """
    plots the image, the noise and the added image + noise besides eachother to
    compare those
    """
    if color:
        cmap = None
    else:
        cmap = "gray"
    image = np.squeeze(image)
    noise = np.squeeze(noise)
    f, axarr = plt.subplots(1,3, figsize=constants.FIG_SIZE_TRIPLE, sharex=True)
    #plt.axis('off')
    axarr[0].imshow(image, vmin=0, vmax=1, cmap=cmap)
    noiseScale = 1./np.max(noise)
    axarr[1].imshow(noise * noiseScale, vmin=-1, vmax=1, cmap=cmap)
    imgPNoise = image+noise
    scaleIN = 1./np.max(imgPNoise)
    axarr[2].imshow(imgPNoise * scaleIN, vmin=0, vmax=1, cmap=cmap)
    if (predictImage != -1):
        axarr[0].set_title('Prediction: ' + str(predictImage))
    else:
        axarr[0].set_title('Image:')
    if (predictNoise != -1):
        axarr[1].set_title('Prediction Noise: ' + str(predictNoise) + "; scaling: " + str(noiseScale)[:4])
    else:
        axarr[1].set_title('Noise, scaling: ' + str(noiseScale)[:4])
    if (predictAdded != -1):
        axarr[2].set_title('Prediction: ' + str(predictAdded))
    else:
        axarr[2].set_title('Image + Noise')
    for axis in axarr:
        axis.set_axis_off()
    if std_noise >= 0:
        plt.figtext(0.52, 0.08, "std: {:.3f}".format(std_noise), ha='center', size=10)

    if prob_image is not None:
        plt.figtext(0.2, 0.08, "probability: {:.0f}\%".format(prob_image*100), ha='center', size=10)
    if prob_adv_ex is not None:
        plt.figtext(0.855, 0.08, "probability: {:.0f}\%".format(prob_adv_ex*100), ha='center', size=10)

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show()