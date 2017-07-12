import keras, cv2
import tensorflow as tf
from keras.models import load_model
from keras import losses
import pandas as pd
import numpy as np
from tqdm import tqdm #tqdm is a library to display a progress bar
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from plot_image_noise import show_img_noise
#from generate_adv_ex import get_gradient
import scipy.optimize, scipy.stats

#instanciate the session at the loading of this module. Could perhaps lead to problems later? But instanciating the
#session in a function doesn't work for unknown reasons.

from keras import backend as K

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

def get_inv_gradient(model, images, results):
    """
    returns the gradient of 1/(1-x0) where x0 is the result of the softmax from results
    """
    sess = K.get_session()
    images = np.array(images, dtype="float32")
    assert len(images.shape) == 4
    assert len(results.shape) == 2
    indices_gather = []
    for i, res in enumerate(results):
        indices_gather.append([i, np.argmax(results)])
    # define gradient operator
    grad_op = tf.gradients(1/(tf.constant([1.001], shape=[1], dtype="float32") - tf.gather_nd(model.outputs[0], indices_gather)),
                           model.inputs[0])[0]


    # retrieve the input placeholder (the image)
    graph_def = tf.get_default_graph().as_graph_def()
    input_placeholder = tf.get_default_graph().get_tensor_by_name(graph_def.node[0].name + ":0")

    # compute the gradient
    grad = sess.run(grad_op, feed_dict={input_placeholder: np.array(images), K.learning_phase(): 0})
    return grad

def create_inv_loss_func_for_minimize(model):
    """
    returns 1/(1+1e-6-x0)
    """
    def loss_func(image, target):
        #print("e")
        prediction = np.squeeze(model.predict(np.array([image], dtype="float32"), batch_size=1, verbose=0)).astype("float64")
        return 1/(1.001-prediction[np.argmax(target)]+1e-6)
    return loss_func



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
    data_X = data_X.astype("float32")
    data_X /= 255.0
    data_Y = keras.utils.to_categorical(train[:, 0])
    return data_X, data_Y

def open_data_dogs_cat_float(beg = 0, end = None, rows=128, cols=128, TRAIN_DIR = '../data/dog_vs_cats/train/'):
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
    sess = K.get_session()
    loss = model.loss
    if (type(loss) is str):
        loss = getattr(losses, loss)

    def loss_func(image, target):
        # print("e")
        prediction = model.predict(np.array([image], dtype="float32"), batch_size=1, verbose=0)
        prediction = tf.convert_to_tensor(prediction)
        target = tf.convert_to_tensor(np.array([target], dtype="float32"))
        # print("f")
        return loss(prediction, target).eval(session=sess).astype("float64")

    return loss_func


def run_minimizer(model, images, truePredictions, num_to_predict = 3):
    loss_func = create_loss_func_for_minimize(model)
    c = 1e5
    d = 1
    imgShape = np.shape(images[0])

    for image, truePrediction in zip(images, truePredictions):

        bounds = np.zeros((images[0].size, 2), dtype="float64")
        bounds[:, 1] = 1.0
        bounds[:, 0] -= image.flatten()
        bounds[:, 1] -= image.flatten()
        x0 = (np.random.rand(*imgShape)-image)*0.1

        target_Y = np.zeros(10, dtype = "float32")
        target_Y[num_to_predict] = 1.
        def to_minimize(r):
            r = r.reshape(imgShape)
            res_loss_func = np.exp(d*loss_func(image + r, target_Y)[0])
            res_norm  = c * np.std(r)
            val = res_norm + res_loss_func
            print("min: {:.3f}, {:.3f}, {:.3f}".format(val, res_loss_func, res_norm))
            return val

        def grad(r):
            r = r.reshape(imgShape)
            grad = d*get_gradient(model, np.array([image + r]), np.array([target_Y]))[0].astype("float64")
            if np.sum(np.abs(grad)) == 0:
                print("grad 0")
                #if gradient is zero, add a random gradient to keep going.
                grad = (np.random.random(imgShape)-0.5)*1e-2
            res_loss_func = d * loss_func(image + r, target_Y)[0]
            gradient = np.exp(res_loss_func) * grad
            #gradient = get_inv_gradient(model, np.array([image + r]), np.array([target_Y]))[0].astype("float32")
            #norm_r = vec_abs(r)
            #grad_norm = c*r/norm_r
            grad_norm = c/r.size*r/np.std(r)
            #print("grad: {:.3f}, {:.3f}".format(np.std(gradient), np.std(grad_norm)))
            return (grad_norm + gradient).flatten()

        res_optimize = scipy.optimize.minimize(to_minimize, jac=grad,
                                        # x0=np.zeros(imgShape),
                                        x0=x0, bounds=bounds, #method="TNC",
                                        method="L-BFGS-B",
                                        tol=0.001, options={"maxiter": 100, "eps": 0.01})
        print("Number of iterations: {}".format(res_optimize.nit))
        tempR = res_optimize.x.reshape(imgShape)
        result_image_and_r = model.predict(np.array([image + tempR], dtype="float32"), batch_size=1, verbose=0)
        prediction_adv_ex = np.argmax(result_image_and_r)

        print("std r: {:.4f}, num predicted {}, probability: {:.1f}%".format(np.std(tempR), prediction_adv_ex,
                                                                              np.max(result_image_and_r)*100))
        prediction_im = np.argmax(truePrediction)
        show_img_noise(image, tempR, predictImage=prediction_im, predictAdded=prediction_adv_ex)

def run_minimizer_inv(model, image, truePrediction, c = 1e2, plot=False, x0_factors = [0.1], save_as = False):
    inv_loss_func = create_inv_loss_func_for_minimize(model)
    c = c #put c = 1e3 for dogs vs cats
    d = 1
    p = 100
    imgShape = np.shape(image)

    bounds = np.zeros((image.size, 2), dtype="float64")
    bounds[:, 1] = 1.0
    bounds[:, 0] -= image.flatten()
    bounds[:, 1] -= image.flatten()
    x0 = (np.random.rand(*imgShape)-image)*0.1
    def norm(r):
        return c*(1/r.size*np.sum(np.abs(r)**p))**(1./p)
    def to_minimize(r):
        r = r.reshape(imgShape)
        res_loss_func = d*inv_loss_func(image + r, truePrediction)
        if res_loss_func == np.inf:
            res_loss_func = 1e9
        res_norm = norm(r)
        val = res_norm + res_loss_func
        if plot:
            print("min: {:.3f}, {:.3f}, {:.3f}".format(val, res_loss_func, res_norm))
        return val

    def grad(r):
        r = r.reshape(imgShape)
        gradient = d*get_inv_gradient(model, np.array([image + r]), np.array([truePrediction]))[0].astype("float64")
        if p == 1:
            grad_norm =c/r.size*np.sign(r)/(norm(r)/c)
        else:
            grad_norm = c/r.size*r*np.abs(r)**(p-2)/(norm(r)/c)**(p-1)
        if plot:
            print("grad: {:.7f}, {:.7f}".format(np.std(gradient), np.std(grad_norm)))
        return (grad_norm + gradient).flatten()
    final_res_optimize = None
    for factor_x0 in x0_factors:
        x0 = (np.random.rand(*imgShape)-image)*factor_x0
        res_optimize = scipy.optimize.minimize(to_minimize, jac=grad,
                                        # x0=np.zeros(imgShape),
                                        x0=x0, bounds=bounds, #method="TNC",
                                        method="L-BFGS-B",
                                        tol=0.001, options={"maxiter": 100, "eps": 0.01})
        if final_res_optimize is None or final_res_optimize.fun > res_optimize.fun:
            final_res_optimize = res_optimize


    noise = final_res_optimize.x.reshape(imgShape)
    result_image = model.predict(np.array([image], dtype="float32"), batch_size=1, verbose=0)
    result_adv_ex = model.predict(np.array([image + noise], dtype="float32"), batch_size=1, verbose=0)
    prediction_adv_ex = np.argmax(result_adv_ex)
    image_target = np.argmax(truePrediction)
    prediction_image = np.argmax(result_image)
    if image_target == prediction_image and (not prediction_adv_ex == prediction_image):
        success = True
    else:
        success = False
    if plot:
        print("Number of iterations: {}, {}".format(final_res_optimize.nit, final_res_optimize.nfev))
        print("std r: {:.5f}, num predicted {}, probability: {:.1f}%".format(np.std(noise), prediction_adv_ex,
                                                                             np.max(result_adv_ex) * 100))
        if "cat" in save_as:
            prediction_image = "cat" if prediction_image == 1 else "dog"
            prediction_adv_ex = "cat" if prediction_adv_ex == 1 else "dog"
        show_img_noise(image, noise, predictImage=prediction_image, predictAdded=prediction_adv_ex,
                       save_as = save_as, std_noise = np.std(noise))
    return_dic = {"image_target": image_target,
                  "result_image": result_image,
                  "prediction_image": prediction_image,
                  "result_adv_example": result_adv_ex,
                  "prediction_adv_example": prediction_adv_ex,
                  "minimize_result": final_res_optimize.fun,
                  "std_noise": np.std(noise),
                  "success": success}
    return return_dic


if __name__ == "__main__":

    #model = load_model('../mnist_models/mnist_model2')
    #dataX, dataY = read_data_mnist()

    model = load_model("../keras_model_cat_dogs8")
    dataX, dataY = open_data_dogs_cat_float(end = 20, rows=128, cols=128)
    for i in range(2,5):
        rs = run_minimizer_inv(model, dataX[i], dataY[i], plot=True, c = 1e3,
                               save_as="../figures/adv_example_minimizer_dogs_vs_cats_L100_{}.svg".format(i))
    """
    predicImg = np.argmax(model.predict(np.array([dataX[1]], dtype="float64"), batch_size=1, verbose=0))
    predicNoise = np.argmax(model.predict(np.array([rs[0]], dtype="float64"), batch_size=1, verbose=0))
    predicImgNoise = np.argmax(model.predict(np.array([dataX[1] + rs[0]], dtype="float64"), batch_size=1, verbose=0))
    show_img_noise(dataX[1], rs[0], predicImg, predicNoise, predicImgNoise)
    """