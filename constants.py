FIG_SIZE = (6, 4)
FIG_SIZE_TRIPLE = (4, 1.5)
import matplotlib as mpl

fig_width_pt = 246  # Get this from LaTeX using \showthe\columnwidth normally 426
inches_per_pt = 1.0/72.27   # Convert pt to inch
golden_mean = (5 ** 0.5 - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]

mpl.rcParams["text.latex.preamble"] = [
                                       "\\usepackage{amsmath,amssymb}",
                                "\\usepackage[separate-uncertainty=true]{siunitx}"]
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams["ps.usedistiller"] = "xpdf"
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["font.size"] = 10
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["text.usetex"] = True
mpl.rcParams["figure.figsize"] = fig_size

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})