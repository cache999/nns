import Fn
import numpy as np
import matplotlib.pyplot as plt


# Fn.py
'''
weights = Fn.WeightInitializations.XavierUniformReLu(5, 1000)
assert max(weights) < 0.64
assert min(weights) > -0.64

weights = Fn.WeightInitializations.XavierUniform(5, 1000)
assert max(weights) < 0.45
assert min(weights) > -0.45

relu = Fn.Activations.ReLu(np.array([-3, 3, -2, 2, -1, 1, 0]))
assert (relu == np.array([0, 3, 0, 2, 0, 1, 0])).all()

relu_primes = Fn.Activations.ReLu(np.array([-3, 3, -2, 2, -1, 1, 0]), prime=True)
# print(relu_primes)
'''
'''
softmax = Fn.Activations.Softmax(np.array([5, 8, 3, 5, 2]))
assert np.sum(softmax) == 1

s_prime = Fn.Activations.Softmax(np.array([[5], [8], [3], [5], [2]]), prime=True)
print(s_prime)

s2 = Fn.Activations.Softmax(np.array([5, 8.01, 3, 5, 2]))

y = Fn.OneHot(2, 1)
print(y.dense())
L = Fn.LossFunctions.CrossEntropy(np.array([[0.2], [0.8]]), y)
L_prime = Fn.LossFunctions.CrossEntropy(np.array([[0.2], [0.8]]), y, prime=True)

Lm1 = Fn.LossFunctions.CrossEntropy(np.array([[0.19], [0.81]]), y)
'''

a = np.array([[3], [10], [6], [8]], dtype=np.float)
y = Fn.OneHot(4, 2)
o = Fn.Activations.softmax(a)
L = Fn.LossFunctions.cross_entropy(o, y)
lliszt = [L]
a0, a1, a2, a3 = [a[0, 0]], [a[1, 0]], [a[2, 0]], [a[3, 0]],
for _ in range(1000):
    dLdO = Fn.LossFunctions.cross_entropy(o, y, prime=True)
    dOdA = Fn.Activations.softmax(a, prime=True)
    dLdA = np.dot(dOdA, dLdO)
    a -= dLdA * 0.01
    o = Fn.Activations.softmax(a)
    L = Fn.LossFunctions.cross_entropy(o, y)
    lliszt.append(L)
    a0.append(a[0, 0])
    a1.append(a[1, 0])
    a2.append(a[2, 0])
    a3.append(a[3, 0])

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_xlabel('iter')
ax1.set_ylabel('loss')
ax2.set_xlabel('iter')
ax2.set_ylabel('val')
ax1.plot(range(1000 + 1), lliszt)
ax2.plot(range(1000 + 1), a0)
ax2.plot(range(1000 + 1), a1)
ax2.plot(range(1000 + 1), a2)
ax2.plot(range(1000 + 1), a3)

plt.show()

print('ging')
