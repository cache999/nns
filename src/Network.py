import Layers
import Fn
import numpy as np
from sklearn.datasets import load_iris
import data_normalization
import matplotlib.pyplot as plt


class Network(object):
    def __init__(self, layers: list, input_dim, output_dim):
        # validate that layers arent borken
        self.layers = layers
        assert self.layers[-1].num_neurons == output_dim
        self.layers[0].num_weights_per_neuron = input_dim + 1
        self.input_dim = input_dim
        self.output_dim = output_dim

        # decide number of weights
        for i in range(1, len(self.layers)):
            self.layers[i].num_weights_per_neuron = self.layers[i - 1].num_neurons + 1

        # make matrices n shit
        for l in self.layers:
            l.weights_with_bias = l.weight_initialization(l.num_weights_per_neuron - 1, (l.num_neurons, l.num_weights_per_neuron))
            l.weights_without_bias = l.weights_with_bias[:,:-1]
            l.activations = np.empty((l.num_neurons, 1))
            l.outputs_with_bias = np.empty((l.num_neurons + 1, 1))
            l.outputs_with_bias[-1] = 1
            l.outputs = l.outputs_with_bias[:-1]
            l.delta = np.zeros((l.num_neurons, l.num_weights_per_neuron))

    def predict(self, x):
        # if type(x) == np.ndarray:
        o = np.concatenate((x, [[1]]))  # add bias
        for l in self.layers:
            np.matmul(l.weights_with_bias, o, out=l.activations)
            l.outputs_with_bias[:-1] = l.activation(l.activations.copy(), prime=False)
            o = l.outputs_with_bias
        return self.layers[-1].outputs

    def train(self, x, y, los_function, learning_rate, batch_size=8):
        train_index = 0
        num_batches = x.shape[0] / batch_size
        assert num_batches.is_integer()
        for _ in range(int(num_batches)):

            # predict, calculate loss, calculate loss prime for 1 batch
            sum_los = 0
            for bi in range(batch_size):
                o = self.predict(x[train_index + bi])
                sum_los += los_function(o, y[train_index + bi])
                dlosdo = los_function(o, y[train_index + bi], prime=True)
                doda = self.layers[-1].activation(self.layers[-1].activations, prime=True)
                # do last layer
                self.layers[-1].pear = np.dot(doda, dlosdo)  # my god im going senile i spent 5 minutes verifying this
                self.layers[-1].delta += np.dot(self.layers[-1].pear, self.layers[-2].outputs_with_bias.T)

                # find amount to adjust each weight
                for li in reversed(range(1, len(self.layers)-1)):  # iterate backwards through layerss
                    dlosdo_li = np.dot(self.layers[li + 1].weights_without_bias.T, self.layers[li + 1].pear)
                    # pear is just dlda_li
                    self.layers[li].pear = dlosdo_li * self.layers[li].activation(
                        self.layers[li].activations, prime=True)
                    self.layers[li].delta += np.dot(self.layers[li].pear, self.layers[li - 1].outputs_with_bias.T)

                # first layer, use inps
                dlosdo_0 = np.dot(self.layers[1].weights_without_bias.T, self.layers[1].pear)
                self.layers[0].pear = dlosdo_0 * self.layers[0].activation(self.layers[0].activations, prime=True)
                train_data_with_bias = np.concatenate((x[train_index + bi], [[1]]))
                self.layers[0].delta += np.dot(self.layers[0].pear, train_data_with_bias.T)
            # adjust weights + reset deltas
            for l in self.layers:
                l.weights_with_bias -= l.delta * learning_rate
                l.delta = np.zeros((l.num_neurons, l.num_weights_per_neuron))

            train_index += batch_size


if __name__ == "__main__":
    np.random.seed(1)

    nn = Network([
        Layers.Dense(10, activation=Fn.Activations.relu,
                     weight_initialization=Fn.WeightInitializations.xavier_uniform_relu),
        Layers.Dense(3, activation=Fn.Activations.softmax,
                     weight_initialization=Fn.WeightInitializations.xavier_uniform_relu)
    ], input_dim=4, output_dim=3)

    for layer in nn.layers:
        print(layer)

    d_size = 150
    cut = int(d_size * 0.9)
    x, y = data_normalization.normalize(load_iris())
    y = np.array([Fn.OneHot(3, i) for i in y])
    # shuffle
    p = np.random.permutation(d_size)
    x, y = x[p], y[p]

    train_x, train_y = x[:cut, :, :], y[:cut]
    test_x, test_y = x[cut:], y[cut:]

    losses = []
    los = np.empty(train_x.shape[0])
    for i in range(train_x.shape[0]):
        los[i] = Fn.LossFunctions.cross_entropy(nn.predict(train_x[i]), train_y[i])
    losses.append(np.mean(los))

    num_epoches = 10000
    for epoch in range(num_epoches):
        if epoch % 100 == 0:
            print('Epoch ' + str(epoch) + '/' + str(num_epoches))
        nn.train(train_x, train_y, los_function=Fn.LossFunctions.cross_entropy, learning_rate=0.01, batch_size=1)

        # los! los! los!
        los = np.empty(train_x.shape[0])
        for i in range(train_x.shape[0]):
            los[i] = Fn.LossFunctions.cross_entropy(nn.predict(train_x[i]), train_y[i])
        losses.append(np.mean(los))

    # test model
    los = np.empty(train_x.shape[0])
    for i in range(train_x.shape[0]):
        los[i] = Fn.LossFunctions.cross_entropy(nn.predict(train_x[i]), train_y[i])
    print('Los: ' + str(np.mean(los)))

    fig, ax = plt.subplots()
    ax.plot(range(num_epoches+1), losses)
    ax.set_xlabel('epoches')
    ax.set_ylabel('los')
    ax.set_yscale('log')
    fig.show()