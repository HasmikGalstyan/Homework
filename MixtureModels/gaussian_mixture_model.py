import numpy as np
from scipy.stats import multivariate_normal as mvn
from k_means_clustering import KMeansClustering


class GaussianMixtureModel:
    def __init__(self, k=5, tol=0.000001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def log_likelihood_loss(self, data, pi, m, omega):
        return sum([np.log(sum([pi[k] * mvn(m[k], omega[k], allow_singular=True).pdf(data[n]) for k in range(self.k)]))
                    for n in range(len(data))])

    def gamma(self, data, pi, m, omega):
        return [[pi[k] * mvn(m[k], omega[k], allow_singular=True).pdf(data[n]) for k in range(self.k)] /
                sum([pi[k] * mvn(m[k], omega[k], allow_singular=True).pdf(data[n]) for k in range(self.k)])
                for n in range(len(data))]

    def mean(self, data, gamma):
        return [sum([gamma[n][k] * data[n] for n in range(len(data))]) / sum(gamma)[k] for k in range(self.k)]

    def omega(self, data, gamma, mean):
        return [sum([gamma[n][k] * np.dot((data[n] - mean[k]).reshape(-1, 1), (data[n] - mean[k]).reshape(1, -1))
                     for n in range(len(data))]) / sum(gamma)[k] for k in range(self.k)]

    def pi(self, gamma):
        return sum(gamma) / sum(sum(gamma))

    def train(self, data):
        loss_dif = 1000000
        gmm_kmc = KMeansClustering()
        gmm_kmc.train(data)
        self.myu = gmm_kmc.myu
        cluster = gmm_kmc.predict(data)
        self.cov = [np.dot((data[cluster == k] - self.myu[k]).T,
                           (data[cluster == k] - self.myu[k])) / len(cluster)
                    for k in range(self.k)]
        self.prob = [sum(cluster == k) / len(cluster) for k in range(self.k)]
        loss = self.log_likelihood_loss(data, self.prob, self.myu, self.cov)
        iteration = 0

        while loss_dif > self.tol and iteration < self.max_iter:
            g = self.gamma(data, self.prob, self.myu, self.cov)
            self.myu = self.mean(data, g)
            self.cov = self.omega(data, g, self.myu)
            self.prob = self.pi(g)
            loss_dif = self.log_likelihood_loss(data, self.prob, self.myu, self.cov) - loss
            loss = self.log_likelihood_loss(data, self.prob, self.myu, self.cov)
            iteration += 1

    def predict(self, data):
        if data.size == data.shape[0]:
            return np.argmax(
                [self.prob[k] * mvn(self.myu[k], self.cov[k], allow_singular=True).pdf(data) for k in range(self.k)] /
                sum([self.prob[k] * mvn(self.myu[k], self.cov[k], allow_singular=True).pdf(data) for
                     k in range(self.k)]))
        return np.argmax(self.gamma(data, self.prob, self.myu, self.cov), axis=1)

    def predict_prob(self, data):
        if data.size == data.shape[0]:
            return [self.prob[k] * mvn(self.myu[k], self.cov[k], allow_singular=True).pdf(data) for k in range(self.k)] \
                   / sum([self.prob[k] * mvn(self.myu[k], self.cov[k], allow_singular=True).pdf(data)
                          for k in range(self.k)])
        return self.gamma(data, self.prob, self.myu, self.cov)
