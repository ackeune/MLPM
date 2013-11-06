import numpy
import matplotlib.pyplot as plt

def plot_histograms(X):
    for row in X:
        row = numpy.array(row.T)
        plt.hist(row)
        plt.show()

X = numpy.matrix('1 1 1 2; 2 3 3 4;5 6 10 5')
plot_histograms(X)
