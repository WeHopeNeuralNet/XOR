import numpy as np
from net import NeuralNet

nn = NeuralNet([2, 3, 1])
nn.train(np.array([[1, 0], [0, 0]]), np.array([[1], [0]]))

