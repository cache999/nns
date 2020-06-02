import numpy as np
import matplotlib.pyplot as plt
from SimpleNN import RNN

np.random.seed(0)

# A simple quadratic regression task to test the basic feedforward neural network.
DATASET_SIZE = 100
LEARNING_RATE = 0.01
mse = lambda test, predict: 0.5 * ((test - predict) ** 2)
mse_prime = lambda test, predict: predict - test

def make_data(error=0, data_points=5, upper_x=1000, lower_x=-1000):
    inputs = np.random.uniform(0, 20, (data_points))
    outputs = 1 / (1 + (2.78 ** (-1 * (inputs - 10))))
    # outputs = (m * inputs) + c
    return inputs, outputs


nn = RNN.NN(1, [1, 3, 1], activation=RNN.LReLu)

inputs, outputs = make_data(
                    data_points = DATASET_SIZE, upper_x = 100, lower_x = 0)

'''
train_in = inputs[:int(len(inputs) * 0.9)]
train_out = outputs[:int(len(outputs) * 0.9)]
test_in = inputs[int(len(inputs) * 0.9):]
test_out = outputs[int(len(outputs) * 0.9):]
'''

train_in = inputs[:int(len(inputs) * 0.2)]
train_out = outputs[:int(len(outputs) * 0.2)]
print(train_in, train_out)

for c in range(800):
    for i, _ in enumerate(train_in):
        nn_out = nn.get_result([train_in[i]])
        loss = mse(train_out[i], nn_out)
        print('Loss at data pair ' + str(i + 1) + ' of epoch ' + str(c + 1) + ': ' + str(loss))

        loss_prime = mse_prime(train_out[i], nn_out)
        nn.backpropagate(LEARNING_RATE, mse_prime(train_out[i], nn_out))

# draw data

fig, ax = plt.subplots()
ax.set_xlabel('train')
ax.set_ylabel('result')

x = []
y = []
for i in range(0, 21):
    x.append(i)
    y.append(nn.get_result([i]))

ax.plot(x, y, label='nn')
ax.scatter(train_in, train_out, label='train')

plt.show()