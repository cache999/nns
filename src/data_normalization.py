import numpy as np
from sklearn.datasets import load_iris


def normalize(dataset):
    # make variance = 1, mean = 0
    data = dataset['data']

    vars = np.var(data, axis=0)
    stds = np.sqrt(vars)
    means = np.mean(data, axis=0)

    for i in range(data.shape[1]):
        data[:, i] -= means[i]
        if stds[i] != 0:
            data[:, i] /= stds[i]
    return data[..., np.newaxis], dataset['target']



if __name__ == "__main__":
    # print(load_iris())
    x, y = normalize(load_iris())

    print('arbeiten')

