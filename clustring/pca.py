from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from clustring.helper import readFile
import matplotlib.pylab as plt


def transform_dataset_into_2_d(items: list):
    items = StandardScaler().fit_transform(items)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(items)
    return principalComponents


def plot_dataset(items, types):
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2 component PCA')
    targets = set(types)
    for target in targets:
        indicesToKeep = [i for i in range(len(types)) if types[i] == target]
        plt.scatter(items[indicesToKeep, 0],
                    items[indicesToKeep, 1],
                    s=25)
    plt.legend(targets)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    data_dir = './data/'
    # fname = 'spiral.data'
    fname = input('Enter the name of the data file: ')
    items, types = readFile(data_dir+fname, ',')

    transformed_items = transform_dataset_into_2_d(items)
    plot_dataset(transformed_items, types)
