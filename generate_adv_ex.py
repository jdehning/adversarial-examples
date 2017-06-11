import keras
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_image(img):
    plt.imshow(np.squeeze(img))
    plt.show()

sess = tf.Session()

from keras import backend as K
K.set_session(sess)

train = pd.read_csv("data/train.csv").values
data_X = train[:, 1:].reshape(train.shape[0], 28, 28, 1)
data_X = data_X.astype(float)
data_X /= 255.0

data_Y = keras.utils.to_categorical(train[:, 0])

model = load_model('keras_model1')
print(model.inputs)
print(model.outputs)
image = data_X[3]
result = data_Y[3]
loss =  K.categorical_crossentropy(model.outputs[0], tf.constant(np.array([result]), dtype = "float32"))
grad_op = tf.gradients(loss, model.inputs[0])[0]

graph_def = tf.get_default_graph().as_graph_def()
input_placeholder = tf.get_default_graph().get_tensor_by_name(graph_def.node[0].name + ":0")
grad = sess.run(grad_op, feed_dict={input_placeholder: np.array([image]), K.learning_phase():0})
random_grad = np.random.randint(0,2,(1,28,28,1))*2-1

modified_image = np.clip(np.array([image]) + 0.2* np.sign(grad), 0,1)
plot_image(np.array([image]) + 0.2* np.sign(grad))
plot_image(modified_image)

print(sess.run(model.outputs[0], feed_dict={input_placeholder: np.array([image]) + 0.2* np.sign(grad), K.learning_phase():0}))
print(sess.run(model.outputs[0], feed_dict={input_placeholder: modified_image, K.learning_phase():0}))
print(sess.run(model.outputs[0], feed_dict={input_placeholder: np.array([image]) + 0.2* random_grad, K.learning_phase():0}))


