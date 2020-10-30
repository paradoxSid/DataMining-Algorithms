from clustring.pca import transform_dataset_into_2_d
import numpy as np
from clustring.helper import readFile
import math
from unionfind import unionfind
import matplotlib.pyplot as plt


class Denclue:
    def __init__(self, items, h, xi, epi, max_iter=100):
        self.items = items
        self.h = h
        self.xi = xi
        self.epi = epi
        self.max_iter = max_iter
        self.number_of_items = self.items.shape[0]
        self.dimension_of_item = self.items.shape[1]
        self.clusters = {}
        self.noise_points = []
        self.run_algorithm()

    def k_gauss(self, x):
        k = math.exp(- np.matmul(x.T, x) / (2 * self.h**2))
        k /= (2*math.pi)**(self.dimension_of_item/2)
        return k

    def f_hat(self, x):
        f = 0
        for xi in self.items:
            f += self.k_gauss(x-xi)
        f /= (self.number_of_items * (self.h ** self.dimension_of_item))
        return f

    def delta_f_hat(self, x):
        delta = np.zeros(self.dimension_of_item)
        for xi in self.items:
            delta += self.k_gauss(x-xi) * (xi-x)
        delta /= (self.number_of_items *
                  (self.h ** (self.dimension_of_item+2)))
        return delta

    def find_attractor(self, item):
        x = item
        for i in range(self.max_iter):
            ks = np.array([self.k_gauss(x-xi) for xi in self.items])
            xt = np.dot(ks, self.items) / ks.sum()
            if np.linalg.norm(xt-x) < self.epi:
                break
            x = xt
        return x

    def run_algorithm(self):
        for item in self.items:
            att = self.find_attractor(item)
            if self.f_hat(att) >= self.xi:
                att = tuple(att)
                if att not in self.clusters:
                    self.clusters[att] = []
                self.clusters[att].append(item)
            else:
                self.noise_points.append(item)

        uf = unionfind(len(self.clusters))
        keys = np.array(list(self.clusters.keys()))
        for i in range(keys.shape[0]):
            for j in range(i+1, keys.shape[0]):
                if np.linalg.norm(keys[i]-keys[j]) <= self.epi:
                    uf.unite(i, j)
        clusters_group = uf.groups()
        clusters = {}
        for group in clusters_group:
            cluster_items = []
            for item in group:
                cluster_items.extend(self.clusters[tuple(keys[item])])
            clusters[tuple(keys[group[0]])] = cluster_items
        self.clusters = clusters
        print(f'Found {len(self.clusters)} clusters')

    def plot_f_hat(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x = self.items[:, 0]
        y = self.items[:, 1]
        z = [self.f_hat(x) for x in self.items]
        # xx, yy = np.meshgrid(np.linspace(min(x), max(x), 10),
        #                      np.linspace(min(y), max(y), 10))
        # zz = xx*0+self.xi
        # ax.plot_surface(xx, yy, zz, alpha=0.5)
        # ax.text(min(x), min(y), self.xi, "XI")

        # ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
        ax.scatter(x, y, z, c=z, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x)')
        ax.set_title(f'Estimated density functions h = {self.h}')

        fig.tight_layout()

        # plt.hlines(self.xi, 0, self.number_of_items, 'r', 'dashed')
        plt.show()

    def plot_cluster(self):
        for k, v in self.clusters.items():
            v = np.array(v)
            plt.plot(v[:, 0], v[:, 1], 'o', ms=3)
        n = np.array(self.noise_points)
        if len(n) > 0:
            plt.plot(n[:, 0], n[:, 1], 'kx', ms=3)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'Denclue Algorithm')
        legends = [f'Cluster - {i}' for i in range(len(self.clusters))]
        plt.legend(legends)
        plt.show()


if __name__ == '__main__':
    data_dir = './data/'

    h = 0.5  # Smoothing parameter
    xi = 0.0003  # Minimum probabilty of an attractor
    epi = .5
    fname = 'iris.data'

    defalut = input('Take Default values(y/n): ')
    if defalut == 'n':
        fname = input('Enter the name of the data file: ')
        h = float(input('Enter the value of h: '))
        xi = float(input('Enter the value of xi: '))
        epi = float(input('Enter the value of epi: '))

    items, types = readFile(data_dir+fname, ',')
    transformed_items = transform_dataset_into_2_d(items)

    denclue = Denclue(transformed_items, h, xi, epi)
    denclue.plot_f_hat()
    denclue.plot_cluster()
    # plot_dataset(transformed_items, types)
