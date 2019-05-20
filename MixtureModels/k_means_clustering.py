import numpy as np


class KMeansClustering:

    def __init__(self, k=5, tol=0.000001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def clustering_loss(self, data, m, r):
        return sum([sum([np.dot(r[n][k] * (data[n] - m[k]), (data[n] - m[k]).T) for k in range(self.k)])
                    for n in range(len(data))])

    def indicator(self, data, m):
        r = np.zeros((len(data), self.k))
        for n in range(len(data)):
            r[n][np.argmin([np.dot(data[n] - m[k], (data[n] - m[k]).T) for k in range(self.k)])] = 1
        return r

    def mean(self, data, r):
        return [sum([r[n][k] * data[n] for n in range(len(data))]) / sum([r[n][k] for n in range(len(data))])
                for k in range(self.k)]

    def train(self, data):
        loss_dif = 1000000
        self.myu = data[np.random.choice(range(len(data)), size=self.k, replace=False)]
        r = self.indicator(data, self.myu)
        loss = self.clustering_loss(data, self.myu, r)
        iteration = 0
        while loss_dif > self.tol and iteration < self.max_iter:
            self.myu = self.mean(data, r)
            r = self.indicator(data, self.myu)
            loss_dif = loss - self.clustering_loss(data, self.myu, r)
            iteration += 1
            loss = self.clustering_loss(data, self.myu, r)

    def predict(self, data):
        return np.argmax(self.indicator(data, self.myu), axis=1)
