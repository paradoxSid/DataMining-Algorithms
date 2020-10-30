import numpy as np
from clustring.helper import readFile
import matplotlib.pyplot as plt


class DBScan:
    def __init__(self, items: list, eps: float, min_points: int, plot=False):
        self.items = items
        self.eps = eps
        self.min_points = min_points
        self.clusters = list()
        self.visited = list()
        self.run_algorithm()
        if plot:
            self.plot_clusters()

    def run_algorithm(self):
        for item in self.items:
            if item in self.visited:
                continue
            self.visited.append(item)
            neighbours = self.get_neighbours(item)
            if len(neighbours) >= self.min_points:
                cluster = self.build_cluster(item, neighbours)
                self.clusters.append(cluster)

    def get_neighbours(self, point: list):
        neighbours = list()
        for item in self.items:
            if np.linalg.norm(np.array(point)-item) <= self.eps:
                if item not in neighbours:
                    neighbours.append(item)
        return neighbours

    def build_cluster(self, item: list, neighbours: list):
        cluster = [item]
        for point in neighbours:
            if point in self.visited:
                continue
            self.visited.append(point)
            nbs = self.get_neighbours(point)
            if len(nbs) >= self.min_points:
                neighbours.extend(nbs)
            cluster.append(point)
        return cluster

    def calculate_sum_squared_error(self):
        cluster_mean = [np.mean(c, axis=0) for c in self.clusters]
        current_sum = 0
        for i in range(len(cluster_mean)):
            cluster_sum = 0
            for item in self.clusters[i]:
                cluster_sum += np.linalg.norm(cluster_mean[i]-item)**2
            current_sum += cluster_sum
        return current_sum

    def plot_clusters(self):
        assigned = list()
        for c in self.clusters:
            assigned.extend(c)
            npArray = np.array(c)
            plt.plot(npArray[:, 0], npArray[:, 1], 'o', ms=3)

        unassigned_nodes = []
        for i in items:
            if i not in assigned:
                unassigned_nodes.append(i)
        unassigned_nodes = np.array(unassigned_nodes)

        legend = [f'Cluster - {i}' for i in range(len(self.clusters))]

        if len(unassigned_nodes) > 0:
            plt.plot(unassigned_nodes[:, 0],
                     unassigned_nodes[:, 1], 'kx', ms=3)
            legend.append('Unassigned')

        plt.title(
            f'Clusters for (eps, min_count) = ({self.eps}, {self.min_points})')
        plt.legend(legend)
        plt.xlabel('Sepal length in cm')
        plt.ylabel('Sepal width in cm')

        plt.grid()
        plt.show()

    def get_noise_count(self):
        return len(self.items) - sum([len(i) for i in self.clusters])


if __name__ == '__main__':
    data_dir = './data/'

    fname = 'iris.data'
    epsmin = .1
    epsmax = 1
    min_counts_min = 1
    min_counts_max = 11
    k = 3
    defalut = input('Take Default values(y/n): ')
    if defalut == 'n':
        fname = input('Enter name of the file in the data dir: ')
        epsmin = float(input('Minimum value of epsilon to test(>0): '))
        epsmax = float(input('Maximum value of epsilon to test(>epsmin): '))
        min_counts_min = int(
            input('Minimum value of min_counts to test(>0): '))
        min_counts_max = int(
            input('Maximum value of min_counts to test(>min_counts_min): '))
        k = int(input('Number of clusters: '))

    items, _ = readFile(data_dir+fname, ',')
    eps = np.linspace(epsmin, epsmax, 10)
    min_counts = range(min_counts_min, min_counts_max)

    # ans in the format (ep, min_count, noise, sse)
    ans = (0, float('inf'), float('inf'),  float('inf'), None)
    for ep in eps:
        for min_count in min_counts:
            print(f'Running for (eps, min_count) = ({ep}, {min_count})')
            dbscan = DBScan(items, ep, min_count)

            if len(dbscan.clusters) != k:
                continue

            noise = dbscan.get_noise_count()
            if ans[2] > noise:
                ans = (ep, min_count, noise,
                       dbscan.calculate_sum_squared_error(), dbscan)

    print('Minimum noise for a given k:', ans[2])
    print('Epsilon for a given k:', ans[0])
    print('Minimun count for a given k:', ans[1])
    print('SSE for a given k:', ans[3])
    ans[-1].plot_clusters()
