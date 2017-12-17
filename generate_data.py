import numpy as np


def target_function(x):
    cosh = np.cosh(x)
    other_value = x ** 3 + np.pi * x ** 2 + np.e * np.arcsin(np.tanh(x))
    return np.sum(cosh + other_value, axis=1)


train_x = np.random.random([1000, 3])
train_y = target_function(train_x)
test_x = np.random.random([100, 3])
test_y = target_function(test_x)

np.save("./data/train_x.npy", train_x)
np.save("./data/train_y.npy", train_y)
np.save("./data/test_x.npy", test_x)
np.save("./data/test_y.npy", test_y)