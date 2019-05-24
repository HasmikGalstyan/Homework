import numpy as np
import matplotlib.pyplot as plt


def gaussian_kernel(y):
    return np.exp(-y ** 2 / 2) / np.sqrt(2 * np.pi)

def uniform_kernel(y):
    if -1 < y < 1:
        return 0.5
    return 0

def epanechnikov_kernel(y):
    return max(1 - y ** 2, 0) * 3 / 4


class KernelDensityEstimation:

    def __init__(self, h=3, kernel=gaussian_kernel):
        self.h = h
        self.kernel = kernel

    def train(self, data):
        self.data = data

    def predict(self, x):
        return sum([self.kernel((x - datum) / self.h) for datum in self.data]) / (len(self.data) * self.h)

    def plot(self):
        x = np.arange(min(self.data), max(self.data), 0.1)
        y = np.array([self.predict(i) for i in x])
        plt.plot(x, y)
        plt.ylabel('Density')
        plt.title('Kernel Density Estimate')
        plt.show()
