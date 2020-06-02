import Fn


class Layer(object):
    pass


class Dense(Layer):
    def __init__(self, neurons, weight_initialization=Fn.WeightInitializations.xavier_uniform,
                 activation=Fn.Activations.relu):
        self.activation = activation
        self.num_neurons = neurons
        self.weight_initialization = weight_initialization
        self.weights_with_bias = None
        self.weights_without_bias = None
        self.activations = None
        self.outputs = None
        self.pear = None
        self.delta = None

    def __repr__(self):
        return 'Dense: weights.shape=' + str(self.weights_with_bias.shape) + ', num_neurons=' + str(self.num_neurons)
