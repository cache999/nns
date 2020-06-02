import math

import numpy as np


class Activations(object):
    @staticmethod
    def relu(a, prime=False):
        if not prime:
            negative = a < 0
            a[negative] = 0
            return a

        else:
            positive = a > 0
            derivatives = positive.astype(int)
            return derivatives

    @staticmethod
    def softmax(a, prime=False):
        if not prime:
            pows = np.power(np.e, a)
            denom = np.sum(pows)
            pows /= denom
            return pows

        else:
            dim = a.shape[0]
            # help i dont actually know vector calc
            i = np.identity(dim)
            softmax = Activations.softmax(a)
            a_tile = np.tile(softmax.T, (dim, 1))
            o_tile = np.tile(softmax, (1, dim))
            jacobian = o_tile * (i - a_tile)
            return jacobian

            # to get vector of dL/da's, A'.T * L (A' is the matrix output by this function)


class LossFunctions(object):
    @staticmethod
    def cross_entropy(o, y, prime=False):
        # L = -sum(ylog(o) + (1-y)log(1-o))
        # equivalent to -1( v[y] . v[log(o)] + v[noty] . v[log(1-o)])
        not_o = 1 - o
        if not prime:
            if type(y) == OneHot:
                a = np.log(not_o)
                a[y.hot_index] = np.log(o[y.hot_index])
                a = np.sum(a)
            else:
                a = np.dot(y, np.log(o))
                a += np.dot(1 - y, np.log(not_o))
            return -1 * a

        else:
            # i dont actually know if this is correct?!!!!
            if type(y) == OneHot:
                a = 1 / not_o
                a[y.hot_index] = (-1) / o[y.hot_index]
            else:
                a = (y / o) + ((1 - y) / not_o)  # lolol this is so lazy
            return a

    @staticmethod
    def kl_divergence(o, y, prime=False):
        if not prime:
            log = np.log2(y / o)
            return np.dot(y.T, log)


class WeightInitializations(object):
    @staticmethod
    def xavier_uniform(n, shape):
        bounds = (-1 * (1 / math.sqrt(n)), 1 / math.sqrt(n))
        return np.random.uniform(bounds[0], bounds[1], shape)

    @staticmethod
    def xavier_uniform_relu(n, shape):
        bounds = (-1 * (math.sqrt(2 / n)), math.sqrt(2 / n))
        return np.random.uniform(bounds[0], bounds[1], shape)


class OneHot(object):
    def __init__(self, length, hot_index):
        self.hot_index = hot_index
        self.length = length
        self.dense_rep = None

    def dense(self):
        if self.dense_rep is None:
            self.dense_rep = np.zeros(self.length)
            self.dense_rep[self.hot_index] = 1
        return self.dense_rep

    def __repr__(self):
        return self.hot_index