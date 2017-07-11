import numpy as np
import matplotlib.pyplot as plt
import sys, glob
import constants

def get_percentage_false_class_for_resultset(results):
    """
    extracts the amount of successful adversarial examples devided by the amout of
    original images that were rightly predicted
    """
    count_success = 0
    count_correct_prediction = 0
    for result in results:
        if result["image_target"] == result["prediction_image"]:
            count_correct_prediction += 1
        if result["success"] == True:
            count_success += 1

    return count_success/count_correct_prediction, 

def read_models():
    model_files_cvd = glob.glob("./grad_results/cvd*N1024_f001.npy")
    model_files_mnist = glob.glob("./grad_results/mnist*N1024*.npy")

    results_cvd = []
    results_mnist = []

    for filename in model_files_cvd:
        results_cvd.append(np.load(filename))
        
    for filename in model_files_mnist:
        results_mnist.append(np.load(filename))

    return np.array(results_mnist), np.array(results_cvd)
        
def get_percentage_false_class():
    model_files_cvd = glob.glob("./grad_results/cvd*N1024_f001.npy")
    model_files_mnist = glob.glob("./grad_results/mnist*N1024*.npy")

    percentages_cvd = []
    percentages_mnist = []

    for filename in model_files_cvd:
        cur_result = np.load(filename)
        percentages_cvd.append(get_percentage_false_class_for_resultset(cur_result))

    for filename in model_files_mnist:
        cur_result = np.load(filename)
        percentages_mnist.append(get_percentage_false_class_for_resultset(cur_result))

    return percentages_mnist, percentages_cvd


def plot_percentage_false_class(percentage, xlabels=None):
    fig, ax = plt.subplots()
    ax.plot(xlabels, percentage, "o")
    plt.show()

if __name__ == "__main__":
    mnist, cvd = get_percentage_false_class()
    print(mnist)
    print(cvd)
    plot_percentage_false_class(cvd, np.arange(1,4))


