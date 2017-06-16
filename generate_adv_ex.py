"""
This module generates adv. examples and calculates some statistics. Interesting functions are:

return_accuracy_of_model_for_given_noise_added(model, ratio_of_noise_added = 0.2)
return_min_noise_for_false_classification(model, images, results):
analyse_min_noise(model, num_images_to_analyse)
"""

import keras
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm #tqdm is a library to display a progress bar
import scipy.optimize, scipy.stats

#instanciate the session at the loading of this module. Could perhaps lead to problems later? But instanciating the
#session in a function doesn't work for unknown reasons.
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

def read_data_mnist():
    train = pd.read_csv("data/train.csv").values
    data_X = train[:, 1:].reshape(train.shape[0], 28, 28, 1)
    data_X = data_X.astype(float)
    data_X /= 255.0
    data_Y = keras.utils.to_categorical(train[:, 0])
    return data_X, data_Y

def get_gradient(model, images, results):
    #images and results need to be a list or array of images and results

    images = np.array(images, dtype="float32")
    results = np.array(results, dtype="float32")
    assert len(images.shape) == 4
    assert len(results.shape) == 2
    # define loss function like in the model
    loss = K.categorical_crossentropy(model.outputs[0], tf.constant(results))

    # define gradient operator
    grad_op = tf.gradients(loss, model.inputs[0])[0]

    # retrieve the input placeholder (the image)
    graph_def = tf.get_default_graph().as_graph_def()
    input_placeholder = tf.get_default_graph().get_tensor_by_name(graph_def.node[0].name + ":0")

    # compute the gradient
    grad = sess.run(grad_op, feed_dict={input_placeholder: np.array(images), K.learning_phase(): 0})
    return grad

def get_softmax_results_from_images(model, images):
    images = np.array(images, dtype="float32")
    assert len(images.shape) == 4
    graph_def = tf.get_default_graph().as_graph_def()
    input_placeholder = tf.get_default_graph().get_tensor_by_name(graph_def.node[0].name + ":0")
    softmax = sess.run(model.outputs[0], feed_dict={input_placeholder: images, K.learning_phase(): 0})
    return softmax

def get_max_of_softmax_res(softmax_res):
    return np.argmax(softmax_res, axis = 1)

def plot_image(img):
    plt.imshow(np.squeeze(img), cmap='gray')
    plt.show()

def batch_generator(*arrays, batch_size):
    #returns batches of size batch_size
    iterators = []
    for arr in arrays:
        iterators.append(iter(arr))
    while True:
        ret = [[] for _ in range(len(iterators))]
        for _ in range(batch_size):
            try:
                for i, iterator in enumerate(iterators):
                    ret[i].append(next(iterator))
            except StopIteration:
                return
        map(np.array, ret)
        yield ret


def return_accuracy_of_model_for_given_noise_added(model, ratio_of_noise_added = 0.2):
    """
    returns ratio of images that are not correctly categorised given a certain noise level.
    """

    dataX, dataY = read_data_mnist()
    len_data = len(dataX)
    num_same = 0
    num_diff = 0

    #calculate in batches, in order to speed up the calculations. Don't put much higher than about 1000 for the MNIST
    #dataset, as it already needs about 1.7GB ram.
    batch_size = 1024
    total_iter = int(len_data/batch_size)
    for images, results in tqdm(batch_generator(dataX, dataY, batch_size=batch_size), total=total_iter):
        grad = get_gradient(model, images, results)
        modified_image = np.clip(np.array(images) + ratio_of_noise_added * np.sign(grad), 0, 1)
        softmax_res = get_softmax_results_from_images(model, images=modified_image)

        res_of_mod_image = get_max_of_softmax_res(softmax_res)
        result_norm = get_max_of_softmax_res(results)
        num_incorr = np.sum((result_norm - res_of_mod_image) != 0)

        num_same += len(images)-num_incorr
        num_diff += num_incorr
    print("percent incorrect: {:.2f}%".format(num_diff/(num_diff + num_same)*100))
    return num_diff/(num_diff + num_same)

def return_min_noise_for_false_classification(model, images, results):
    """
    for a given model, images and results, this function finds the minimum noise needed to get a false classification.
    It returns an array of the noises. If no adv example can be found, the noise is composed of np.nan
    """

    n_iter_basinhopping = 1 #put to 1 for fast results, to 10 for better ones
    gradients = get_gradient(model, images, results)
    noise_arr = []

    for i, (images, results, grad) in tqdm(enumerate(batch_generator(images, results, gradients, batch_size = 1)), total=len(images)):
        #grad = np.random.randint(0,2,(1,28,28,1))*2-1
        if np.sum(grad) == 0:
            print("gradient is zero, won't find adv. ex.")
            noise_arr.append(np.full(images[0].shape, np.nan))
            continue

        def modify_single_image(images, noise_ratio):
            return np.clip(np.array(images[0]) + noise_ratio * np.sign(np.array(grad[0])), 0, 1)
        def min_func(noise_ratio):
            if noise_ratio < 0 or noise_ratio > 1:
                return 1e5
            modified_image = [modify_single_image(images, noise_ratio)]
            modified_results = get_softmax_results_from_images(model, images=modified_image)
            num_origin_result = get_max_of_softmax_res(results)[0]

            if get_max_of_softmax_res(modified_results)[0] == get_max_of_softmax_res(results)[0]:
                diff_to_false_class = 0.001
                diff_to_origin = modified_results[0] - modified_results[0][num_origin_result] - diff_to_false_class
                diff_to_origin = np.delete(diff_to_origin, num_origin_result)
                minimize_value = 1 / (np.sum((1 / diff_to_origin) ** 2)) ** (1. / 2) + 1

                return minimize_value
            else:
                return noise_ratio[0]
        def set_bounds_test(x_new, **kwargs):
            if x_new[0] >= 0 and x_new[0] <= 1:
                return True
            else:
                return False

        res_optimization = scipy.optimize.basinhopping(min_func, [0.5], T = 0.05, stepsize = 0.2,
                                                       accept_test = set_bounds_test, niter = n_iter_basinhopping,
                                                       minimizer_kwargs = {"method":"L-BFGS-B",
                                                                           "options": {"eps":0.001, "ftol":1e-5}})
        #print(i, res_optimization.x[0])
        if not min_func(res_optimization.x) < 1:
            print("Couldn't find adv. example")
            noise_arr.append(np.full(images[0].shape, np.nan))
        else:
            noise_arr.append(modify_single_image(images, res_optimization.x[0])-images[0])
    return np.array(noise_arr)

def analyse_min_noise(model, num_images_to_analyse):
    """
    returns the mean standard deviation and the standard error of the mean of the minimium noise calculated with the
    gradient method for the mnist dataset. For only a random number of num_images_to_analyse the adversarial example is
    calculated.
    """

    dataX, dataY = read_data_mnist()
    index_chosen = np.random.choice(range(len(dataX)), size=num_images_to_analyse, replace=False)
    noises = return_min_noise_for_false_classification(model, dataX[index_chosen], dataY[index_chosen])
    std_arr = np.nanstd(noises, (1,2,3))
    return np.nanmean(std_arr), scipy.stats.sem(std_arr, nan_policy="omit")

#random_grad = np.random.randint(0,2,(1,28,28,1))*2-1

if __name__ == "__main__":
    model = load_model('keras_model1')
    #return_accuracy_of_model_for_given_noise_added(model)
    print(analyse_min_noise(model, 10))




