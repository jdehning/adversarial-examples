import keras
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data():
    train = pd.read_csv("data/train.csv").values
    data_X = train[:, 1:].reshape(train.shape[0], 28, 28, 1)
    data_X = data_X.astype(float)
    data_X /= 255.0
    data_Y = keras.utils.to_categorical(train[:, 0])
    return data_X, data_Y

def get_gradient(model, image, result):
    sess = tf.Session()
    from keras import backend as K
    K.set_session(sess)
    # define loss function like in the model
    loss = K.categorical_crossentropy(model.outputs[0], tf.constant(np.array([result]), dtype="float32"))

    # define gradient operator
    grad_op = tf.gradients(loss, model.inputs[0])[0]

    # retrieve the input placeholder (the image)
    graph_def = tf.get_default_graph().as_graph_def()
    input_placeholder = tf.get_default_graph().get_tensor_by_name(graph_def.node[0].name + ":0")

    # compute the gradient
    grad = sess.run(grad_op, feed_dict={input_placeholder: np.array([image]), K.learning_phase(): 0})
    sess.close()
    return grad

def get_softmax_results_from_images(model, images):
    assert len(images.shape) == 4
    sess = tf.Session()
    graph_def = tf.get_default_graph().as_graph_def()
    input_placeholder = tf.get_default_graph().get_tensor_by_name(graph_def.node[0].name + ":0")
    softmax = sess.run(model.outputs[0], feed_dict={input_placeholder: image, K.learning_phase(): 0})
    return softmax

def get_max_of_softmax_res(softmax_res):
    return np.argmax(softmax_res)

def plot_image(img):
    plt.imshow(np.squeeze(img))
    plt.show()

def return_accuracy_of_model_for_given_noise_added(model):
    dataX, dataY = read_data()
    for image, result in zip(dataX, dataY):
        print(image)
        print(image.shape)
        grad = get_gradient(model, image, result)
        modified_image = np.clip(np.array([image]) + 0.2 * grad, 0, 1)
        softmax_res = get_softmax_results_from_images(model, images=modified_image)
        res_of_mod_image = get_max_of_softmax_res(softmax_res)
        if res_of_mod_image == softmax_res:
            print("same")
        else:
            print("different")






#random_grad = np.random.randint(0,2,(1,28,28,1))*2-1

if __name__ == "__main__":
    model = load_model('keras_model1')
    return_accuracy_of_model_for_given_noise_added(model)



