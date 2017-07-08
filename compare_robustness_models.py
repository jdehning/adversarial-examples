
from multiprocessing import Pool
import numpy as np
import pickle, os, cv2
"""
def compare_cats_dogs_minimizer():
    model_nums = [8,9,10]
    models = []
    for num in model_nums:
        models = [keras.models.load_model("keras_model_cat_dogs{}".format(i)) for i in model_nums]
        dataX, dataY = open_data_dogs_cat_float(end=2, rows=128, cols=128)
        allresults_dic = {8:[], 9:[], 10:[]}
        for image, target in zip(dataX, dataY):
            prediction_correct = True
            for model in models:
                result_image = model.predict(np.array([image], dtype="float32"), batch_size=1, verbose=0)
                if not np.argmax(result_image) == np.argmax(target):
                    prediction_correct = False
            if not prediction_correct:
                continue
            for model, number in zip(models, model_nums):
                allresults_dic[number].append(run_minimizer_inv(model, image, target, c = 1e3))

    for num in model_nums:
        for res_dic in allresults_dic[num]:
            print(res_dic["std_noise"])
"""
def open_data_dogs_cat_float(beg = 0, end = None, rows=128, cols=128, TRAIN_DIR = './data/dog_vs_cats/train/'):
    train_dogs = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'dog' in i][beg:end]
    train_cats = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'cat' in i][beg:end]
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

def compare_cats_dogs_minimizer(args):
    model_num, dataX, dataY = args
    from minimize.generate_minimize_ex import run_minimizer_inv
    from keras import backend as K
    import tensorflow as tf
    import keras
    res_arr = []
    for i, (image,target) in enumerate((zip(dataX, dataY))):
        K.clear_session()
        #sess = K.get_session()
        #sess.close()
        tf.reset_default_graph()
        K.set_session(tf.Session())
        model = keras.models.load_model("keras_model_cat_dogs{}".format(model_num))
        print("Hi")
        res_dic = run_minimizer_inv(model, image, target, c = 1e3)
        res_dic["index"] = i
        res_arr.append(res_dic)
        print(model_num,i, "{:.4f}, {:.3f}".format(res_dic["std_noise"], res_dic["minimize_result"]))
        if i%4 == 0:
            pickle.dump(res_arr, open("adv_exampled_cats_dogs_model{}_{}".format(model_num, i), "wb"))
            if i >= 20:
                os.remove("adv_exampled_cats_dogs_model{}_{}".format(model_num, i-20))

def compare_dics(filenames):
    res_arrays = [pickle.load(open(file, "rb")) for file in filenames]
    plot_arrays = [[] for _ in range(len(res_arrays))]
    for i_image in range(len(res_arrays[2])):
        save = True
        values = []
        for i_model in range(len(res_arrays)):
            res_dic = res_arrays[i_model][i_image]
            if not res_dic["success"] == True:
                save = False
                break
            values.append(res_dic["minimize_result"])
        if save == True:
            for i, val in enumerate(values):
                plot_arrays[i].append(val)




if __name__ == "__main__":
    p = Pool(3)
    dataX, dataY = open_data_dogs_cat_float(end=1000, rows=128, cols=128, TRAIN_DIR="./data/dog_vs_cats/train/")
    p.map(compare_cats_dogs_minimizer, [(i, dataX, dataY) for i in [8,9,10]])