
from multiprocessing import Pool, Manager
import numpy as np
import pickle, os, cv2, glob, scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import copy, time



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

def get_last_filename_mnist(model_num):
    potential_filenames = []
    dir = "./"
    for filename in os.listdir(dir):
        if filename.find("adv_exampled_mnist_model{}".format(model_num)) >= 0:
            potential_filenames.append(os.path.join(dir, filename))
    potential_filenames.sort(key=lambda x: int(x.split("_")[-1]))
    if not potential_filenames == []:
        return potential_filenames[-1]
    else:
        return []

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

def compare_mnist_minimizer(args):
    model_num, dataX, dataY = args
    last_filename = get_last_filename_mnist(model_num)
    if last_filename == []:
        res_arr = []
    else:
        res_arr = pickle.load(open(last_filename, "rb"))
    index = len(res_arr)
    if index >= 1000:
        return 1

    from minimize.generate_minimize_ex import run_minimizer_inv
    from keras import backend as K
    import tensorflow as tf
    import keras
    for image,target in zip(dataX[index: index+20], dataY[index: index+20]):
        K.clear_session()
        tf.reset_default_graph()
        K.set_session(tf.Session())
        model = keras.models.load_model("./mnist_models/mnist_model{}".format(model_num))
        res_dic = run_minimizer_inv(model, image, target, c = 1e2, x0_factors=[0.1,0.4])
        res_dic["index"] = index
        res_arr.append(res_dic)
        print(model_num, index, "{:.4f}, {:.3f}".format(res_dic["std_noise"], res_dic["minimize_result"]))
        index += 1
    pickle.dump(res_arr, open("adv_exampled_mnist_model{}_{}".format(model_num, index), "wb"))
    if index > 20:
        os.remove("adv_exampled_mnist_model{}_{}".format(model_num, index - 20))

def read_data_mnist():
    train = pd.read_csv("./data/train.csv").values
    data_X = train[:, 1:].reshape(train.shape[0], 28, 28, 1)
    data_X = data_X.astype("float32")
    data_X /= 255.0
    data_Y = to_categorical(train[:, 0])
    return data_X, data_Y

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def compare_dics(cats = False):
    if cats:
        model_numbers = [9,8,10]
        res_arrays = [pickle.load(open(get_last_filename_cat_dogs(model_number), "rb")) for model_number in model_numbers]
    else:
        model_numbers = [1,2,3]
        res_arrays = [pickle.load(open(get_last_filename_mnist(model_number), "rb")) for model_number in
                      model_numbers]
    plot_arrays = [[] for _ in range(len(res_arrays))]
    for i_image in range(len(res_arrays[2])):
        save = True
        values = []
        for i_model in range(len(res_arrays)):
            res_dic = res_arrays[i_model][i_image]
            if (not res_dic["success"] == True) or res_dic["minimize_result"] > 15:
                save = False
                break
            values.append(res_dic["minimize_result"])
        if save == True:
            for i, val in enumerate(values):
                plot_arrays[i].append(val)

    min = 1.
    if cats:
        max = 8.
    else:
        max = 12
    colors = ["r", "b", "g"]
    if cats:
        labels = ["shallow: 13 layers", "medium: 14 layers", "deep: 15 layers"]
    else:
        labels = ["shallow: 6 layers", "medium: 7 layers", "deep: 8 layers"]
    f, ax = plt.subplots()
    for i, plot_array in enumerate(plot_arrays):
        bw = (max - min) / np.sqrt(len(plot_array))
        print(plot_array)
        plot_kernel_distribution(plot_array, ax, label=labels[i], bw=bw,
                                 min=min, max=max, color = colors[i])
    plt.xlim(min, max)
    plt.xlabel("Minimizer value")
    plt.tight_layout()
    if cats:
        plt.savefig("figures/plot_cats_vs_dogs_robustness_minimizer.pdf")
    else:
        plt.savefig("figures/plot_mnist_robustness_minimizer.pdf")
    plt.show()


def read_models_gradient(cats):
    if cats:
        model_files = glob.glob("./grad_results/cvd*N1024_f0003.npy")
        model_files =  np.append(model_files[1:], [model_files[0]])
    else:
        model_files = glob.glob("./grad_results/mnist*N25000_f02.npy")

    results = []

    for filename in sorted(model_files, key = lambda x: int(x.split(".")[1].split("_")[-3][5:])):
        results.append(np.load(filename))

    return np.array(results)

def compare_dics_gradient(cats = False):
    res_arrays = read_models_gradient(cats)

    plot_arrays = [[] for _ in range(len(res_arrays))]

    num_adv_ex = np.array([0,0,0])
    num_total = 0
    for i_image in range(len(res_arrays[0])):
        save = True
        values = []
        for i_model in range(len(res_arrays)):
            res_dic = res_arrays[i_model][i_image]
            if res_dic["std_noise"] == 0 or (not res_dic["image_target"] == res_dic["prediction_image"]):
                save = False
                break
            values.append(res_dic["result_adv_example"][res_dic["image_target"]])

        if save == True:
            num_total += 1
            for i_model, val in enumerate(values):
                res_dic = res_arrays[i_model][i_image]
                if not res_dic["prediction_image"] == res_dic["prediction_adv_example"]:
                    num_adv_ex[i_model] += 1
                plot_arrays[i_model].append(val)

    print(num_adv_ex/num_total)
    p1 = num_adv_ex[0]/num_total
    p2 = num_adv_ex[1]/num_total
    def calc_z(p1, p2, n):
        p = (p1+p2)/2.
        z = (p1 - p2)/np.sqrt(p*(1-p)*2/n)
        return z
    z = calc_z(p1, p2, num_total)
    print(z)
    print(2*(1 - scipy.stats.norm.cdf(abs(calc_z(p1, p2, num_total)))))

    #min = -np.log(1)
    min = 0
    if cats:
        max = 1
    else:
        #max = -np.log(0.999)
        max = 1
    colors = ["r", "b", "g"]
    if cats:
        labels = ["shallow: 13 layers", "medium: 14 layers", "deep: 15 layers"]
    else:
        labels = ["shallow: 6 layers", "medium: 7 layers", "deep: 8 layers"]
    f, ax = plt.subplots()
    for i, plot_array in enumerate(plot_arrays):
        bw = (max - min) / np.sqrt(len(plot_array))
        print(plot_array)
        plot_kernel_distribution(plot_array, ax, label=labels[i], bw=bw,
                                 min=min, max=max, color = colors[i])
    plt.xlim(min, max)
    plt.xlabel("Minimizer value")
    plt.tight_layout()
    if cats:
        pass
        #plt.savefig("figures/plot_cats_vs_dogs_robustness_minimizer.pdf")
    else:
        pass
        #plt.savefig("figures/plot_mnist_robustness_minimizer.pdf")
    plt.show()



def plot_kernel_distribution(x, ax, min, max, bw, label = "", color = "b"):
    try:
        from statsmodels.nonparametric.kernel_density import KDEMultivariate
        import_succeded = True
    except ImportError:
        import_succeded = False
    x = np.array(x)
    x = x[~np.isnan(x)]
    bw_bars = bw * np.ones_like(x)
    bars = [min]
    for width in bw_bars:
        bars += [bars[-1] + width * 1.2]
    ax.hist(x, bars, normed=1, facecolor=color, alpha=0.2)
    if import_succeded: #plot a kernel distribution
        #print(x.shape)
        #print(bw)
        kde = KDEMultivariate(np.array([x]).transpose(), bw=[bw], var_type=["c" ])
        x_grid = np.linspace(min, max, 1000)
        #print(x_grid.shape)
        pdf = np.array(kde.pdf(x_grid))
        ax.plot(x_grid, pdf, linewidth=3, alpha=0.9, label='{}, {} pts'.format(label, len(x)), color = color)
    ax.legend(loc='upper right')

def manager_for_cats_vs_dogs():
    p = Pool(1, maxtasksperchild = 1)
    manager = Manager()
    dataX, dataY = open_data_dogs_cat_float(end=1000, rows=128, cols=128, TRAIN_DIR="./data/dog_vs_cats/train/")
    dataX = manager.list(dataX)
    dataY = manager.list(dataY)
    for _ in range(100):
        res = p.map_async(compare_cats_dogs_minimizer, [(i, dataX, dataY) for i in [8, 9, 10]])
    for _ in range(3):
        p.map_async(time.sleep, [3600*40 for i in range(2)])
    p.close()
    print("closed")
    p.join()
    print("joined")

def manager_for_mnist():
    p = Pool(3, maxtasksperchild = 1)
    manager = Manager()
    dataX, dataY = read_data_mnist()
    dataX = manager.list(dataX[:1000])
    dataY = manager.list(dataY)[:1000]
    for _ in range(50):
        res = p.map_async(compare_mnist_minimizer, [(i, dataX, dataY) for i in [1, 2, 3]])
        print(res.get())
    for _ in range(3):
        p.map_async(time.sleep, [3600*40 for i in range(2)])
    p.close()
    print("closed")
    p.join()
    print("joined")

if __name__ == "__main__":
    #manager_for_cats_vs_dogs()
    #manager_for_mnist()
    compare_dics(cats=False)
    #compare_dics_gradient(cats = True)
