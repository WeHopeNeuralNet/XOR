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
    
    def train(self, x_datas, y_datas, max_iterations=1000, error_threshold=0.05):
        iteration_count = 0
        while iteration_count < max_iterations:
            for x_data, y_data in six.moves.zip(x_datas, y_datas):
                output = self.foward(x_data)
                self.back(y_data)
            
            if self.check():
                break
            
            iteration_count += 1
    
    def foward(self, x_data):
#         self.first_layer = x_data
#         
#         for i in range(self.shape[1]):
#             unit_sum = 0.0
#             for j in range(self.shape[0]):
#                 unit_sum += self.first_layer[j] * self.weights[1][i]
#         
#         for i in range(self.shape[2]):
#             self.weights[i].dot(x_data)
#         
#         for i in range(self.shape[3]):
#             unit_sum = 0.0
#             for j in range(self.shape[0]):
#                 unit_sum += self.first_layer[j] * self.weights[1][i]
        
        
        
        
    def predict(self, layer):
        self.weights[layer].dot()
        
        
        
        