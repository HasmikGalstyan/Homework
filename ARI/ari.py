import numpy as np
from sklearn.linear_model import LinearRegression


class ARI:
    def __init__(self, s=[7], d=1, p=1):
        self.s = s
        self.d = d
        self.p = p
        self.lr = LinearRegression(fit_intercept=False)

    def train(self, data):
        self.data = [data]
        for i in self.s:
            data = data[i:] - data[:-i]
            self.data.append(data)
        for i in range(self.d):
            data = np.diff(data)
            self.data.append(data)

    def predict(self, num):
        for i in range(num):
            self.lr.fit([self.data[-1][i:i - self.p:-1] for i in range(self.p, len(self.data[-1]) - 1)],
                        self.data[-1][self.p + 1:])
            self.data[-1] = np.append(self.data[-1], self.lr.predict(
                [self.data[-1][i:i - self.p:-1] for i in range(self.p + 1, len(self.data[-1]))])[-1])
            for j in range(2, len(self.data) + 1):
                self.data[-j] = np.append(self.data[-j],
                                          self.data[-j + 1][-1] + self.data[-j][len(self.data[-j + 1]) - 1])
        return self.data[0][-num:]
