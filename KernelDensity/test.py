import numpy as np
from kernel_density import *

if __name__=='__main__':

    data = np.concatenate((np.random.normal(4,1, 500), np.random.normal(10,2,500)))

    kde = KernelDensityEstimation(2, kernel = epanechnikov_kernel)
    kde.train(data)
    print(kde.predict(10))
    kde.plot()
