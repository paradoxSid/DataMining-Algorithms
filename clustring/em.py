from clustring.pca import transform_dataset_into_2_d
import numpy as np
from scipy.stats import multivariate_normal as mvn
from clustring.helper import readFile
import random
import matplotlib.pyplot as plt


class EMAlgorithm:
    def __init__(self, items, k, max_iter=100, eps=1e-7):
        self.items = items
        self.number_of_clusters = k
        self.number_of_items = self.items.shape[0]
        self.dimension_of_item = self.items.shape[1]
        self.max_iter = max_iter
        self.eps = eps
        # Mean of i cluster with in d dimensions
        self.means = np.random.rand(k, self.dimension_of_item)
        # Sigma of i cluster with in d dimensions
        self.sigma = np.random.rand(k, self.dimension_of_item)
        # Fraction of items came from i cluster
        self.pi = np.random.rand(k)
        self.run_algorithm()
        self.plot()

    def run_algorithm(self):
        log_likelihood = 0
        for t in range(self.max_iter):
            bis = np.zeros((self.number_of_clusters, self.number_of_items))

            for i in range(self.number_of_clusters):
                gnormal = mvn(self.means[i], self.sigma[i],
                              allow_singular=True).pdf(self.items)
                bis[i, :] = self.pi[i] * gnormal

            bis /= bis.sum(0)

            # Recalculating pis, means and sigmas
            self.pi = bis.sum(1)/self.number_of_items
            self.means = np.dot(bis, self.items) / bis.sum(1)[:, None]
            self.sigma = np.zeros(
                (self.number_of_clusters, self.dimension_of_item, self.dimension_of_item))
            for i in range(self.number_of_clusters):
                ys = self.items - self.means[i, :]
                temp = (
                    bis[i, :, None, None] * np.matmul(ys[:, :, None], ys[:, None, :])).sum(axis=0)
                self.sigma[i] = temp
            self.sigma /= bis.sum(axis=1)[:, None, None]

            # Convergence criteria
            log_likelihood_new = 0
            for pi, mu, sigma in zip(self.pi, self.means, self.sigma):
                log_likelihood_new += pi*mvn(mu, sigma).pdf(self.items)
            log_likelihood_new = np.log(log_likelihood_new).sum()

            if np.abs(log_likelihood_new - log_likelihood) < self.eps:
                break
            log_likelihood = log_likelihood_new

    def plot(self):
        intervals = 101
        ys = np.linspace(-8, 8, intervals)
        X, Y = np.meshgrid(ys, ys)
        _ys = np.vstack([X.ravel(), Y.ravel()]).T

        z = np.zeros(len(_ys))
        for pi, mu, sigma in zip(self.pi, self.means, self.sigma):
            z += pi*mvn(mu, sigma).pdf(_ys)
        z = z.reshape((intervals, intervals))

        ax = plt.subplot(111)
        plt.scatter(self.items[:, 0], self.items[:, 1], alpha=0.2)
        plt.contour(X, Y, z)
        plt.axis([-6, 6, -6, 6])

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'EM Algorithm')

        plt.grid()
        plt.show()


if __name__ == '__main__':
    data_dir = './data/'
    # fname = 'iris.data'
    fname = input('Enter the name of the data file: ')
    k = int(input('Enter the number of clusters: '))
    items, types = readFile(data_dir+fname, ',')
    transformed_items = transform_dataset_into_2_d(items)

    EMAlgorithm(transformed_items, k)
    # plot_dataset(transformed_items, types)
