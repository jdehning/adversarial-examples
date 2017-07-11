
from multiprocessing import Pool, Manager
import numpy as np
import pickle, os, cv2
import matplotlib.pyplot as plt
import copy
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

def get_last_filename_cat_dogs(model_num):
    potential_filenames = []
    for filename in os.listdir("./"):
        if filename.find("adv_exampled_cats_dogs_model{}".format(model_num)) >= 0:
            potential_filenames.append(filename)
    potential_filenames.sort(key=lambda x: int(x.split("_")[-1]))
    return potential_filenames[-1]

def compare_cats_dogs_minimizer(args):
    model_num, dataX, dataY = args
    res_arr = pickle.load(open(get_last_filename_cat_dogs(model_num), "rb"))
    index = len(res_arr)
    if index >= 1000:
        return 1

    from minimize.generate_minimize_ex import run_minimizer_inv
    from keras import backend as K
    import tensorflow as tf
    import keras

    """
    data_X_local = copy.deepcopy()
    data_Y_local = copy.deepcopy(dataY[index, index+10])
    del dataX, dataY
    """
    for image,target in zip(dataX[index: index+10], dataY[index: index+10]):
        K.clear_session()
        tf.reset_default_graph()
        K.set_session(tf.Session())
        model = keras.models.load_model("keras_model_cat_dogs{}".format(model_num))
        res_dic = run_minimizer_inv(model, image, target, c = 1e3)
        res_dic["index"] = index
        res_arr.append(res_dic)
        print(model_num, index, "{:.4f}, {:.3f}".format(res_dic["std_noise"], res_dic["minimize_result"]))
        index += 1
    pickle.dump(res_arr, open("adv_exampled_cats_dogs_model{}_{}".format(model_num, index), "wb"))
    os.remove("adv_exampled_cats_dogs_model{}_{}".format(model_num, index - 10))


def compare_dics():
    model_numbers = [8,9,10]
    res_arrays = [pickle.load(open(get_last_filename_cat_dogs(model_number), "rb")) for model_number in model_numbers]
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

    min = 1.
    max = 8.
    colors = ["r", "b", "g"]
    labels = ["average depth", "shallow", "deep"]
    f, ax = plt.subplots()
    for i, plot_array in enumerate(plot_arrays):
        bw = (max - min) / np.sqrt(len(plot_array))
        print(plot_array)
        plot_kernel_distribution(plot_array, ax, label=labels[i], bw=bw,
                                 min=min, max=max, color = colors[i])
    plt.xlim(min, max)
    plt.xlabel("Minimizer value")
    plt.tight_layout()
    plt.show()


def plot_kernel_distribution(x, ax, min, max, bw, label = "", color = "b"):
    try:
        from statsmodels.nonparametric.kernel_density import KDEMultivariate
        import_succeded = True
    except ImportError:
        import_succeded = False
    x = np.array(x)
    x = x[~np.isnan(x)]
    bw = bw * np.ones_like(x)
    bars = [min]
    for width in bw:
        bars += [bars[-1] + width * 1.2]
    ax.hist(x, bars, normed=1, facecolor=color, alpha=0.2)
    if import_succeded: #plot a kernel distribution
        print(x.shape)
        print(bw)
        kde = KDEMultivariate(x, var_type="c")
        x_grid = np.linspace(min, max, 1000)
        print(x_grid.shape)
        pdf = np.array(kde.pdf(x_grid))
        ax.plot(x_grid, pdf, linewidth=3, alpha=0.9, label='{}, {} pts'.format(label, len(x)), color = color)
    ax.legend(loc='upper right')

if __name__ == "__main__":
    """
    p = Pool(3)
    manager = Manager()
    dataX, dataY = open_data_dogs_cat_float(end=1000, rows=128, cols=128, TRAIN_DIR="./data/dog_vs_cats/train/")
    dataX = manager.list(dataX)
    dataY = manager.list(dataY)
    for _ in range(100):
        res = p.map_async(compare_cats_dogs_minimizer, [(i, dataX, dataY) for i in [8,9,10]])
    p.close()
    print("closed")
    p.join()
    print("joined")
    """
    compare_dics()
