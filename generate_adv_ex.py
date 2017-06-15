import keras
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#tqdm is a library to
from tqdm import tqdm

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
        yield ret


def return_accuracy_of_model_for_given_noise_added(model):
    dataX, dataY = read_data_mnist()
    len_data = len(dataX)
    num_same = 0
    num_diff = 0
    ratio_of_noise_added = 0.2

    #calculate in batches, in order to speed up the calculations
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



#random_grad = np.random.randint(0,2,(1,28,28,1))*2-1

if __name__ == "__main__":
    model = load_model('keras_model1')
    return_accuracy_of_model_for_given_noise_added(model)





