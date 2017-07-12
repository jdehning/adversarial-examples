import numpy as np
import matplotlib.pyplot as plt
import sys, glob
import constants
from statsmodels.stats.proportion import proportion_confint

def get_percentage_false_class_for_resultset(results):
    """
    extracts the amount of successful adversarial examples devided by the amout of
    original images that were rightly predicted
    """
    count_success = 0
    count_correct_prediction = 0
    for result in results:
        if result["image_target"] == result["prediction_image"] and result["std_noise"] != 0:
            count_correct_prediction += 1
        if result["success"] == True:
            count_success += 1

    error = stats.proportion.proportion_confint(count_success, count_correct_prediction, 0.05)

    return np.array([count_success/count_correct_prediction, error])

def read_models():
    """
    read the modelfiles for the mnist and cats vs dogs dataset and returns them in ordered
    manner
    """
    model_files_cvd = np.sort(glob.glob("./grad_results/cvd*N1024_f0003.npy"))
    model_files_mnist = np.sort(glob.glob("./grad_results/mnist*N25000_f02.npy"))

    model_files_cvd = np.append(model_files_cvd[1:], [model_files_cvd[0]])

    results_cvd = []
    results_mnist = []

    for filename in model_files_cvd:
        results_cvd.append(np.load(filename))
        
    for filename in model_files_mnist:
        results_mnist.append(np.load(filename))

    return np.array(results_mnist), np.array(results_cvd)
        
def get_percentage_false_class(arr_of_results):
    """
    returns the percentage of false classification for the given resultsets
    produced by different models. Only images useable in all set are being
    considered

    arr_of_results: array returned by read_models
    """

    count_success = np.zeros_like(arr_of_results[:,0], dtype=float)
    count_correct_prediction = 0

    for i in range(len(arr_of_results[0])):
        use = True
        for result in arr_of_results[:,i]:
            if result["image_target"] != result["prediction_image"] or result["std_noise"] == 0:
                use = False
        if use:
            count_correct_prediction += 1
            i2 =  0
            for result in arr_of_results[:,i]:
                if result["success"]:
                    count_success[i2] += 1
                i2 += 1


    errors = proportion_confint(count_success, count_correct_prediction)
    count_success = count_success/count_correct_prediction
    errors = np.array(errors)

    errors[0] = np.abs(count_success - errors[0])
    errors[1] = np.abs(count_success - errors[1])

    return count_success, errors


def plot_percentage_false_class(percentage, errors=None, xlabels=None, c=0.2):
    fig, ax = plt.subplots()
    x = range(len(percentage))
    if np.all(errors != None):
        ax.bar(x, percentage, width=0.5, yerr=errors, alpha=0.8, ecolor="b")
        ewidith = 0.2
        for i in range(len(x)):
            ax.plot([x[i]-ewidith, x[i]+ewidith], [percentage[i] - errors[0][i], percentage[i] - errors[0][i]], "b")
            ax.plot([x[i]-ewidith, x[i]+ewidith], [percentage[i] + errors[1][i], percentage[i] + errors[1][i]], "b")
    else:
        ax.bar(x, percentage)
    plt.tight_layout()
    ax.set_ylim(0, 1)
    ax.set_xlim(min(x) - 0.51, max(x) + 0.51)
    ax.set_xlabel("number of layers")
    ax.set_ylabel("misclassification rate")
    #ax.set_title("Successrate of adversarial examples created with the gradient method and c = "+str(c))
    #ax.grid()
    ax.legend(["$\epsilon=" + str(c) + "$"])
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    return fig, ax

if __name__ == "__main__":
    mnist, cvd = read_models()
    mnist_success, mnist_errors = get_percentage_false_class(mnist)
    cvd_success, cvd_errors = get_percentage_false_class(cvd)


    #print(mnist)
    #print(cvd)
    x_labels_mnist = np.array(["6", "7", "8"])
    x_labels_cvd = np.array(["13", "14", "15"])
    plot_percentage_false_class(mnist_success, mnist_errors, x_labels_mnist, 0.2)
    plt.savefig("figures/mnist_grad_misclassificationrate.pdf")
    plot_percentage_false_class(cvd_success, cvd_errors, x_labels_cvd, 0.03)
    plt.savefig("figures/cvd_grad_misclassificationrate.pdf")



