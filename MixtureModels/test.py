from sklearn import datasets
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import numpy as np
from gaussian_mixture_model import GaussianMixtureModel
from k_means_clustering import KMeansClustering
import time

blobs = datasets.make_blobs(n_samples=500,centers=5,cluster_std=1.4,random_state=110)

blobs_data = blobs[0]
blobs_cluster = blobs[1]

plt.subplot(1,3,1)
plt.scatter(blobs_data[:,0], blobs_data[:,1], c=blobs_cluster, alpha=0.5)
plt.title('Original Clusters')

kmc_start = time.time()
kmc = KMeansClustering()
kmc.train(blobs_data)
kmc_pred = kmc.predict(blobs_data)
kmc_stop = time.time()
print("KMeansClustering runtime in secounds is {}".format(kmc_stop-kmc_start))

plt.subplot(1,3,2)
plt.scatter(x=blobs_data[:,0], y=blobs_data[:,1], c=kmc_pred, alpha=0.5)
plt.title('KMeansClustering')

gmm_start = time.time()
gmm = GaussianMixtureModel()
gmm.train(blobs_data)
gmm_pred = gmm.predict(blobs_data)
gmm_stop = time.time()
print("GaussianMixtureModel runtime in secounds is {}".format(gmm_stop-gmm_start))

plt.subplot(1,3,3)
plt.scatter(x=blobs_data[:,0], y=blobs_data[:,1], c=gmm_pred, alpha=0.5)
plt.title('GaussianMixtureModel')

plt.show()
