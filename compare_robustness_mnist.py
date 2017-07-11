import numpy as np
import matplotlib.pyplot as plt
import glob

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
    

def compare_grad_dics():
    mnist_filenames = glob.glob("grad_results/cvd*f001.npy")
    data_to_plot = []
    for fname in mnist_filenames:
        cur_data = np.load(fname)
        cur_data_to_plot = []
        for val in cur_data:
            if (val['success'] == True):
                cur_data_to_plot.append(val['std_noise'])

        data_to_plot.append(cur_data_to_plot)
    data_to_plot = np.array(data_to_plot)

    minimum = np.min(np.min(data_to_plot))
    maximum = np.max(np.max(data_to_plot))

    colors = ["r", "b", "g"]
    labels = ["average depth", "shallow", "deep"]
    f, ax = plt.subplots()
    for i, plot_array in enumerate(data_to_plot):
        bw = (maximum - minimum) / np.sqrt(len(plot_array))
        #print(plot_array)
        plot_kernel_distribution(plot_array, ax, label=labels[i], bw=bw,
                                 min=minimum, max=maximum, color = colors[i])
    ax.legend(loc='upper right')
    plt.xlim(minimum, maximum)
    plt.xlabel("Standard deviation noise")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_grad_dics()