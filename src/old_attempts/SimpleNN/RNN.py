import numpy as np
np.random.seed(0)


class RNN(object):
    def __init__(self, hidden_x, hidden_y):
        self.hidden = np.zeros((hidden_x, hidden_y))

    def step(self, x: list, last: bool = False):
        """

        :param x: One-hot representation of a letter.
        :param last: Only returns if last = True.
        :return: None if last = False, list (one hot representation of vowel phonemes) if last = True
        """
        pass


class NN(object):
    def __init__(self, num_inputs, layer_sizes: list, activation):
        assert layer_sizes[-1] == 1
        self.net = [Layer(num_inputs, layer_sizes[0])]
        self.activation = activation
        self.x = None
        for i in range(1, len(layer_sizes)):
            self.net.append(Layer(layer_sizes[i - 1], layer_sizes[i]))

    def get_result(self, x: list) -> int:
        self.x = x
        self.net[0].calculate_values_from_array(x, self.activation)
        for i in range(1, len(self.net)):
            self.net[i].calculate_values_from_layer(self.net[i - 1], self.activation)
        return self.net[-1].neurons[0].activated

    # oh GoD bAckPropagatIon

    def backpropagate(self, learning_rate, loss_prime):
        """

        :param learning_rate: int, learning rate. How big a backpropagation step should be.
        :param loss_prime: derivative of the loss function.
        :return: None
        """
        if self.activation == LReLu:
            activation_prime = LReLu_prime
        elif self.activation == ReLu:
            activation_prime = ReLu_prime
        else:
            raise ModuleNotFoundError('ur gay')

        top_neuron = self.net[-1].neurons[0]
        top_neuron.gourd_looking_thing = loss_prime * activation_prime(top_neuron.weight_net)
        for w_i, _ in enumerate(top_neuron.weights):
            if w_i == len(top_neuron.weights) - 1:  # is last weight aka bias
                coefficient = 1
            else:
                coefficient = self.net[-2].neurons[w_i].activated
            # backpropagate!
            '''
            if top_neuron.gourd_looking_thing * coefficient < 0:
                top_neuron.weights[w_i] += learning_rate
            else:
                top_neuron.weights[w_i] -= learning_rate
            '''
            top_neuron.weights[w_i] -= top_neuron.gourd_looking_thing * coefficient * learning_rate

        for l_i in reversed(range(len(self.net) - 1)):
            upper_layer_gourds = [n.gourd_looking_thing for n in self.net[l_i + 1].neurons]
            for n_i, neuron in enumerate(self.net[l_i].neurons):
                # calculate the gourd looking thing
                upper_layer_weights = [n.weights[n_i] for n in self.net[l_i + 1].neurons]
                neuron.gourd_looking_thing = np.dot(upper_layer_gourds, upper_layer_weights) * activation_prime(
                    neuron.weight_net)
                # bp the weights
                for w_i, _ in enumerate(neuron.weights):
                    if w_i == len(neuron.weights) - 1:  # bias
                        coefficient = 1
                    else:
                        if l_i == 0:  # first layer, coef = x.
                            coefficient = self.x[w_i]
                        else:
                            coefficient = self.net[l_i - 1].neurons[w_i].activated
                    # backpropagate!
                    '''
                    if neuron.gourd_looking_thing * coefficient < 0:
                        neuron.weights[w_i] += learning_rate
                    else:
                        neuron.weights[w_i] -= learning_rate
                    '''
                    neuron.weights[w_i] -= neuron.gourd_looking_thing * coefficient * learning_rate

        '''
        desired = dEdO * learning_rate * -1
        for l_i, _ in reversed(range(len(self.net))): #traverse from last layer to first
            for neuron in self.net[l_i]:
                dEdNet =
                for i, _ in enumerate(neuron.weights):
                    # BP each weight so it would produce the desired change.
                    # Also set bp_goals for recursive BP.
                    if i == len(neuron.weights) - 1:
                        dEdNet = dEdO * (ReLu_prime(neuron.weight_net))
                    else:
                        dEdNet =
                    if i == len(neuron.weights - 1): # last weight, or the bias
                        dEdW = dEdNet
                        neuron.weights[i] += dEdW * learning_rate * -1

                    else:
                        if l_i == 0: # first layer, BP using inputs instead.
                            dEdW = dEdNet * self.x[i]
                            neuron.weights[i] += dEdW * learning_rate * -1

                        else:
                            # find corresponding neuron in the previous layer which is the coefficient of the weight
                            dEdW = dEdNet * self.net[l_i - 1].neurons[i]
                            neuron.weights[i] += dEdW * learning_rate * -1

                            # set bp_goals.
                            neuron.bp_goal[i] = dEdNet * neuron.weights[i]
        '''


class Layer(object):
    def __init__(self, previous_num, num_neurons):
        self.previous_num = previous_num
        self.num_neurons = num_neurons
        self.neurons = [Neuron(previous_num + 1) for _ in range(num_neurons)]

    def calculate_values_from_array(self, array: list, activation):
        array = array.copy()
        array.append(1)
        array = np.array(array).T
        for neuron in self.neurons:
            neuron.weight_net = np.dot(neuron.weights, array)
            neuron.activated = activation(neuron.weight_net)

    def calculate_values_from_layer(self, layer, activation):
        array = np.array([n.activated for n in layer.neurons] + [1]).T
        for neuron in self.neurons:
            neuron.weight_net = np.dot(neuron.weights, array)
            neuron.activated = activation(neuron.weight_net)

    def __repr__(self):
        return 'Layer with ' + str(self.num_neurons) + ' neurons, each with ' + str(self.previous_num) + ' + 1 weights.'


class Neuron(object):
    def __init__(self, num_weights):
        self.weights = np.random.uniform(-0.5, 0.5, num_weights)
        self.gourd_looking_thing = None  # the weird delta thing used in backprop
        self.weight_net = None
        self.activated = None


def ReLu(o):
    if o < 0:
        return 0
    return o


def ReLu_prime(x):
    if x <= 0:
        return 0
    return 1


def LReLu(o):
    if o < 0:
        return o * 0.01
    return o


def LReLu_prime(o):
    if o < 0:
        return 0.01
    if o == 0:
        return 0
    return 1
