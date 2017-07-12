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
import pickle, os, sys, cv2
from plot_image_noise import show_img_noise

#instanciate the session at the loading of this module. Could perhaps lead to problems later? But instanciating the
#session in a function doesn't work for unknown reasons.
from keras import backend as K

#
# reading cats vs dogs data
#


def open_data_dogs_cat_float(beg = 0, end = None, rows=128, cols=128, TRAIN_DIR = './data/dog_vs_cats/train/'):
    train_dogs = [TRAIN_DIR + i for i in np.sort(os.listdir(TRAIN_DIR)) if 'dog' in i][beg:end]
    train_cats = [TRAIN_DIR + i for i in np.sort(os.listdir(TRAIN_DIR)) if 'cat' in i][beg:end]
    images = train_dogs + train_cats
    data_Y = np.array([[1,0] for _ in range(len(train_dogs))] + [[0,1] for _ in range(len(train_cats))])
    data_X = prep_data_cat_data_float(images, rows=rows, cols=cols)
    del images
    return data_X, data_Y

def read_image(file_path, rows, cols):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (rows, cols), interpolation=cv2.INTER_CUBIC)[...,::-1]


def prep_data_cat_data_float(images, rows=128, cols=128):
    count = len(images)
    data = np.ndarray((count, rows, cols, 3), dtype="float32")

    for i, image_file in enumerate(images):
        image = read_image(image_file, rows, cols)/255.
        data[i] = image
        if (i+1) % 1000 == 0: print('Processed {} of {}'.format(i+1, count))

    return data

#
# reading mnist data
#
def read_data_mnist():
    train = pd.read_csv("data/train.csv").values
    data_X = train[:, 1:].reshape(train.shape[0], 28, 28, 1)
    data_X = data_X.astype(float)
    data_X /= 255.0
    data_Y = keras.utils.to_categorical(train[:, 0])
    return np.array(data_X), np.array(data_Y)

#
# creating adversarial examples using the gradient method
#
def get_gradient(model, images, results):
    sess = K.get_session()
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
    sess = K.get_session()
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


def return_accuracy_of_model_for_given_noise_added(model, images, truePredictions, indicies, ratio_of_noise_added = 0.2):
    """
    returns ratio of images that are not correctly categorised given a certain noise level.
    """

    len_data = len(images)
    num_same = 0
    num_diff = 0

    #calculate in batches, in order to speed up the calculations. Don't put much higher than about 1000 for the MNIST
    #dataset, as it already needs about 1.7GB ram.
    batch_size = np.min([len_data, 128])
    total_iter = int(len_data/batch_size)
    dics = []
    for images, results in tqdm(batch_generator(images, truePredictions, batch_size=batch_size), total=total_iter):
        images = np.array(images)
        results = np.array(results)

        grad = get_gradient(model, images, results)
        modified_images = np.clip(np.array(images) + ratio_of_noise_added * np.sign(grad), 0, 1)
        

        result_images = model.predict(images.astype(dtype="float32"), batch_size=batch_size, verbose=0)
        result_adv_exs = model.predict(modified_images.astype(dtype="float32"), batch_size=batch_size, verbose=0)
        for i in range(len(images)):
            image_target = np.argmax(results[i])
            prediction_image = np.argmax(result_images[i])
            prediction_adv_ex = np.argmax(result_adv_exs[i])
            noise = modified_images[i] - images[i]
            dics.append({"image_target": image_target,
                         "result_image": result_images[i],
                         "prediction_image": prediction_image,
                         "result_adv_example": result_adv_exs[i],
                         "prediction_adv_example": prediction_adv_ex,
                         "constant": ratio_of_noise_added,
                         "std_noise": np.std(noise),
                         "mean_noise": np.mean(noise),
                         "index": indicies[i],
                         "success": image_target == prediction_image and image_target != prediction_adv_ex})

    return np.array(dics)

def return_min_noise_for_false_classification(model, images, results, indicies):
    """
    for a given model, images and results, this function finds the minimum noise needed to get a false classification.
    It returns an array of the noises. If no adv example can be found, the noise is composed of np.nan
    """

    n_iter_basinhopping = 10 #put to 1 for fast results, to 10 for better ones
    gradients = get_gradient(model, images, results)
    noise_arr = []

    batch_size = min(len(images), 128)
    result_images = model.predict(images.astype(dtype="float32"), batch_size=batch_size, verbose=0)
        

    for i in tqdm(range(len(images))):
        grad = gradients[i]
    #for i, (images, results, grad) in tqdm(enumerate(batch_generator(images, results, gradients, batch_size = 1)), total=len(images)):
        #grad = np.random.randint(0,2,(1,28,28,1))*2-1
        if np.sum(grad) == 0:
            print("gradient is zero, won't find adv. ex.")
            noise_arr.append({"success": False})
            continue

        def modify_single_image(image, noise_ratio):
            return np.clip(np.array(image) + noise_ratio * np.sign(np.array(grad)), 0, 1)
        def min_func(noise_ratio):
            if noise_ratio[0] < 0 or noise_ratio[0] > 1:
                return 1e5
            modified_image = [modify_single_image(images[i], noise_ratio[0])]
            modified_results = get_softmax_results_from_images(model, images=modified_image)
            num_origin_result = get_max_of_softmax_res(np.array([results[i]]))[0]

            if get_max_of_softmax_res(modified_results)[0] == get_max_of_softmax_res(np.array([results[i]]))[0]:
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
        modified_image = modify_single_image(images[i], res_optimization.x[0])
        noise = modified_image - images[i]
        image_target = np.argmax(results[i])
        prediction_image = np.argmax(result_images[i])
        result_noise = model.predict(np.array([noise], dtype="float32"), batch_size=1, verbose=0)
        prediction_noise = np.argmax(result_noise)
        result_adv_ex = model.predict(np.array([modified_image], dtype="float32"), batch_size=1, verbose=0)
        prediction_adv_ex = np.argmax(result_adv_ex)
        if not min_func(res_optimization.x) < 1:
            print("Couldn't find adv. example")
            noise_arr.append({"image": np.full(images[i].shape, np.nan),
                              "noise": np.full(images[i].shape, np.nan),
                              "image_target": image_target,
                              "result_image": result_images[i],
                              "result_noise": result_noise,
                              "prediction_noise": prediction_noise,
                              "prediction_image": prediction_image,
                              "result_adv_example": result_adv_ex,
                              "prediction_adv_example": prediction_adv_ex,
                              "constant": res_optimization.x[0],
                              "std_noise": np.std(noise),
                              "mean_noise": np.mean(noise),
                              "index": indicies[i],
                              "success": False })
        else:
            noise_arr.append({"image": images[i],
                              "noise": np.array(modify_single_image(images[i], res_optimization.x[0])-images[i]),
                              "image_target": image_target,
                              "result_image": result_images[i],
                              "prediction_image": prediction_image,
                              "result_noise": result_noise,
                              "prediction_noise": prediction_noise,
                              "result_adv_example": result_adv_ex,
                              "prediction_adv_example": prediction_adv_ex,
                              "constant": res_optimization.x[0],
                              "std_noise": np.std(noise),
                              "mean_noise": np.mean(noise),
                              "index": indicies[i],
                              "success": image_target==prediction_image and image_target!=prediction_adv_ex })
    return np.array(noise_arr)

def analyse_min_noise(model, num_images_to_analyse):
    """
    returns the mean standard deviation and the standard error of the mean of the minimium noise calculated with the
    gradient method for the mnist dataset. For only a random number of num_images_to_analyse the adversarial example is
    calculated.
    """

    dataX, dataY = read_data_mnist()
    index_chosen = np.random.choice(range(len(dataX)), size=num_images_to_analyse, replace=False)
    noises_info = return_min_noise_for_false_classification(model, dataX[index_chosen], dataY[index_chosen])
    noises = []
    for dicNoise in noises_info:
        noises.append(dicNoise["image"])
    noises = np.array(noises)
    std_arr = np.nanstd(noises, (1,2,3))
    return np.nanmean(std_arr), scipy.stats.sem(std_arr, nan_policy="omit")

def run_gradient(model, image, truePrediction, num_images=1000):
    noise = return_min_noise_for_false_classification(model, np.array([image]), np.array([truePrediction]))
    noise = noise[0]
    result_image = model.predict(np.array([image], dtype="float32"), batch_size=1, verbose=0)
    result_adv_ex = model.predict(np.array([image + noise], dtype="float32"), batch_size=1, verbose=0)
    prediction_adv_ex = np.argmax(result_adv_ex)
    image_target = np.argmax(truePrediction)
    prediction_image = np.argmax(result_image)
    
    return_dic = {"image_target": image,
                  "result_image": result_image,
                  "prediction_image": prediction_image,
                  "result_adv_example": result_adv_ex,
                  "prediction_adv_example": prediction_adv_ex,
                  "noise": noise,
                  "minimize_result": final_res_optimize.fun,
                  "std_noise": np.std(noise)}

def run_classification_const_c(amountImgs, factor_noise, model_number=1, cvd=False):
    if cvd:
        saveName = "grad_results/cvd_model" + str(model_number) + "_N" + str(amountImgs) + "_f" + str(factor_noise).replace(".", "")
        modelName = "keras_model_cat_dogs" + str(model_number)
        dataX, dataY = open_data_dogs_cat_float(beg = 0, end = amountImgs)
    else:
        saveName = "grad_results/mnist_model" + str(model_number) + "_N" + str(amountImgs) + "_f" + str(factor_noise).replace(".", "")
        modelName = "./mnist_models/mnist_model" + str(model_number)
        dataX, dataY = read_data_mnist()

    print("Generating adversarial examples using the gadient method using a factor of " + str(factor_noise))
    print("processing model: " + modelName)
    model = load_model(modelName)
    results = return_accuracy_of_model_for_given_noise_added(model, np.array(dataX[:amountImgs]), np.array(dataY[:amountImgs]), np.arange(0, amountImgs), factor_noise)
    count_success = 0
    count_correct_prediction = 0
    for result in results:
        if result["image_target"] == result["prediction_image"] and result["std_noise"] != 0:
            count_correct_prediction += 1
        if result["success"] == True:
            count_success += 1
    print("amount misclassification of adv ex: " + str(count_success/count_correct_prediction) + "%")
    print("amount correct prediction: " + str(count_correct_prediction/amountImgs) + "%")
    np.save(saveName, results)

def create_image_for_talk(amountImgs=1, model_number=1, cvd=False, save=False):
    if cvd:
        saveName = "figures/cvd_model" + str(model_number) + "_I{}_f{}.svg"
        modelName = "keras_model_cat_dogs" + str(model_number)
        dataX, dataY = open_data_dogs_cat_float(beg = 0, end = amountImgs)
        prediction_arr = ["dog", "cat"]
    else:
        saveName = "figures/mnist_model" + str(model_number) + "_I{}_f{}.svg"
        modelName = "./mnist_models/mnist_model" + str(model_number)
        dataX, dataY = read_data_mnist()   
        prediction_arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    model = load_model(modelName)

    if amountImgs == 1:
        noises = return_min_noise_for_false_classification(model, np.array([dataX[0]]), np.array([dataY[0]]), np.array([0]))
    else:
        noises = return_min_noise_for_false_classification(model, dataX[:amountImgs], dataY[:amountImgs], np.arange(amountImgs))


    for i in range(len(noises)):
        result = noises[i]
        if result["success"] == False:
            continue
        print("image number: " + str(result["index"]))
        print("std: " + str(result["std_noise"]))
        if(save):
            show_img_noise(result["image"], result["noise"], predictImage=prediction_arr[result["prediction_image"]], predictAdded=prediction_arr[result["prediction_adv_example"]], std_noise=result["std_noise"], color = False, save_as = saveName.format(i, str(result["constant"]).replace(".", "")[:4] ))
        else:
            show_img_noise(result["image"], result["noise"], predictImage=prediction_arr[result["prediction_image"]], predictAdded=prediction_arr[result["prediction_adv_example"]], std_noise=result["std_noise"], color = False, save_as=False)

    return


def create_image_for_report(amountImgs=1, model_number=1, cvd=False, save=False):
    if cvd:
        saveName = "report/figures/cvd_model" + str(model_number) + "_I{}_f{}.pdf"
        modelName = "keras_model_cat_dogs" + str(model_number)
        dataX, dataY = open_data_dogs_cat_float(beg=0, end=amountImgs)
        prediction_arr = ["dog", "cat"]
    else:
        saveName = "report/figures/mnist_model" + str(model_number) + "_I{}_f{}.pdf"
        modelName = "./mnist_models/mnist_model" + str(model_number)
        dataX, dataY = read_data_mnist()
        prediction_arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    model = load_model(modelName)

    if amountImgs == 1:
        noises = return_min_noise_for_false_classification(model, np.array([dataX[0]]), np.array([dataY[0]]),
                                                           np.array([0]))
    else:
        noises = return_min_noise_for_false_classification(model, dataX[:amountImgs], dataY[:amountImgs],
                                                           np.arange(amountImgs))

    for i in range(len(noises)):
        result = noises[i]
        if result["success"] == False:
            continue
        print("image number: " + str(result["index"]))
        print("std: " + str(result["std_noise"]))
        if (save):
            show_img_noise(result["image"], result["noise"], predictImage=prediction_arr[result["prediction_image"]],
                           predictAdded=prediction_arr[result["prediction_adv_example"]], std_noise=result["std_noise"],
                           color=False, save_as=saveName.format(i, str(result["constant"]).replace(".", "")[:4]),
                           prob_image=np.max(result["result_image"]), prob_adv_ex=np.max(result["result_adv_example"]))
        else:
            show_img_noise(result["image"], result["noise"], predictImage=prediction_arr[result["prediction_image"]],
                           predictAdded=prediction_arr[result["prediction_adv_example"]], std_noise=result["std_noise"],
                           color=False, save_as=False,
                           prob_image=np.max(result["result_image"]), prob_adv_ex=np.max(result["result_adv_example"]))


#random_grad = np.random.randint(0,2,(1,28,28,1))*2-1

if __name__ == "__main__":

    #
    # for generating the classification dataset
    #
    #if (len(sys.argv) != 3):
    #   print("Usage: python3 generate_adv_ex.py factor_noise model_number")
    #    sys.exit(1)

    #factor_noise = float(sys.argv[1])
    #model_number = str(sys.argv[2])
    #amountImgs = 25000
    #run_classification_const_c(amountImgs, factor_noise, model_number, False)

    
    #
    # for generating actual images of the adv examples
    #

    #create_image_for_talk(amountImgs=5, model_number=2, cvd=False, save=True)
    #create_image_for_talk(amountImgs=5, model_number=9, cvd=True, save=True)
    create_image_for_report(amountImgs=1, model_number=8, cvd=True, save=True)
