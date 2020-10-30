import numpy as np
import matplotlib.pyplot as plt
from clustring.helper import readFile


class KMeans:
    def __init__(self, items, k, max_iter=300, plot=False):
        idx = np.random.randint(len(items), size=k)

        cluster_items = [set() for _ in range(k)]
        items_assigned_cluster = [-1 for _ in range(len(items))]

        for i in range(k):
            cluster_items[i].add(idx[i])
            items_assigned_cluster[idx[i]] = i

        cluster_sum = np.array([items[list(cluster_items[i])[0]]
                                for i in range(k)])
        cluster_mean = [cluster_sum[i]/len(cluster_items[i]) for i in range(k)]

        self.items = items
        self.cluster_items = cluster_items
        self.items_assigned_cluster = items_assigned_cluster
        self.cluster_sum = cluster_sum
        self.cluster_mean = cluster_mean
        self.max_iter = max_iter
        self.plot = plot

        self.run_algorithm()

    def run_algorithm(self):
        for t in range(self.max_iter):
            changes = False
            if self.plot:
                self.plot_2d_graph(t)

            for i in range(len(items)):
                cluster_index = self.assign_cluster(
                    self.cluster_mean, self.items[i])
                if self.items_assigned_cluster[i] == cluster_index:
                    continue
                changes = True
                prev = self.items_assigned_cluster[i]
                if prev != -1:
                    self.cluster_items[prev].remove(i)
                    self.cluster_sum[prev] -= items[i]
                    if len(self.cluster_items[prev]) == 0:
                        self.cluster_mean[prev] = self.cluster_sum[prev]
                    else:
                        self.cluster_mean[prev] = self.cluster_sum[prev] / \
                            len(self.cluster_items[prev])

                self.items_assigned_cluster[i] = cluster_index
                self.cluster_items[cluster_index].add(i)
                self.cluster_sum[cluster_index] += items[i]
                self.cluster_mean[cluster_index] = self.cluster_sum[cluster_index] / \
                    len(self.cluster_items[cluster_index])
            if not changes:
                break

    def plot_2d_graph(self, itr):
        for cluster_item in self.cluster_items:
            plt.plot([items[i][0] for i in cluster_item],
                     [items[i][1] for i in cluster_item],
                     'o', ms=3)
        plt.plot([mean[0] for mean in self.cluster_mean],
                 [mean[1] for mean in self.cluster_mean],
                 '*', ms=5)
        not_assigned = []
        for i in range(len(self.items_assigned_cluster)):
            if self.items_assigned_cluster[i] == -1:
                not_assigned.append(i)
        plt.plot([items[i][0] for i in not_assigned],
                 [items[i][1] for i in not_assigned],
                 'ko', ms=3)
        plt.title(f'k-means iteration: {itr}')
        legends = [f'Cluster - {i}' for i in range(k)]
        legends.append('Mean')
        legends.append('Not Assigned')
        plt.legend(legends)
        plt.show()

    def assign_cluster(self, means, item):
        min_dis = float('inf')
        index = -1
        for i in range(len(means)):
            dis = np.linalg.norm(means[i]-item)
            if min_dis > dis:
                min_dis = dis
                index = i

        return index

    def calculate_sum_squared_error(self):
        current_sum = 0
        for i in range(len(self.cluster_mean)):
            cluster_sum = 0
            for item_index in self.cluster_items[i]:
                cluster_sum += np.linalg.norm(
                    self.cluster_mean[i]-self.items[item_index])**2
            current_sum += cluster_sum
        return current_sum


def get_optimal_k(sum_squared_errors, range_k):
    np_range = np.array(range_k)[:, None]
    np_sse = np.array(sum_squared_errors)[:, None]
    points = np.concatenate((np_range, np_sse), axis=1)

    dis = np.abs(np.cross(points[0]-points[-1], points-points[0])) / \
        np.linalg.norm(points[0]-points[-1])
    max_index = dis.argmax(axis=0)+1
    return max_index


if __name__ == '__main__':
    data_dir = './data/'
    fname = 'iris.data'
    kmin = 1
    kmax = 11
    defalut = input('Take Default values(y/n): ')
    if defalut == 'n':
        fname = input('Enter name of the file in the data dir: ')
        kmin = int(input('Minimum value of k to test(>0): '))
        kmax = int(input('Maximum value of k to test(>kmin): '))

    items, _ = readFile(data_dir+fname, ',')

    sum_squared_errors = []
    for k in range(kmin, kmax):
        print(f'Running for k = {k}')
        kmean = KMeans(items, k)
        sum_squared_errors.append(kmean.calculate_sum_squared_error())

    optimal_k = get_optimal_k(sum_squared_errors, range(kmin, kmax))
    print(f'Optimal K in range {kmin} to {kmax} = {optimal_k}')

    plt.plot(sum_squared_errors, 'o-')
    plt.annotate('Optimal K',
                 xy=(optimal_k, sum_squared_errors[optimal_k]),
                 xycoords='data',
                 xytext=(25, 50), textcoords='offset points', size=10,
                 arrowprops=dict(
                     arrowstyle="->",
                     connectionstyle="angle,angleA=0,angleB=-90,rad=10"
                 ))
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('k vs SSE plot')
    plt.grid()
    plt.show()
