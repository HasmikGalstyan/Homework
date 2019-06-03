import numpy as np


class LRTimeSeries:
    def __init__(self, window, d=1):
        self.window = window
        self.d = d

    def train(self, data):
        self.data = data
        matrix = np.array([np.append(np.array([1]), np.array(data[i - self.d:i - self.window:-self.d]))
                           for i in range(self.window, len(data), self.d)])
        y = np.array(data[self.window:len(data):self.d])
        self.beta = np.dot(np.linalg.inv(np.dot(matrix.T, matrix)), np.dot(matrix.T, y))

    def predict(self):
        pred_matrix = np.array([np.append(np.array([1]), np.array(self.data[i:i - self.window + self.d:-self.d]))
                                for i in range(self.window, len(self.data), self.d)])
        return np.dot(pred_matrix, self.beta)[-1]