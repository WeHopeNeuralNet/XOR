import numpy as np
import six

class NeuralNet:
    
    def __init__(self, shape):
        self.shape = shape
        self.init_weights()
    
    def init_weights(self):
        self.weights = {}
        for layer, cur_layer in enumerate(self.shape[1:]):
            prev_layer = self.shape[layer] + 1
            self.weights[layer + 1] = np.random.rand(cur_layer, prev_layer)
    
    def predict(self):
        