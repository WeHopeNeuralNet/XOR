import numpy as np
import six
import math

class NeuralNet:
    
    def __init__(self, shape):
        self.shape = shape
        self.init_weights_and_gradients()
        
    def init_weights_and_gradients(self):
        self.weights, self.gradients = {}, {}
        for layer, cur_layer in enumerate(self.shape[1:]):
            prev_layer = self.shape[layer] + 1
            self.weights[layer + 1] = np.random.rand(cur_layer, prev_layer)
            self.gradients[layer + 1] = np.zeros([cur_layer, prev_layer])
    
    def train(self, x_datas, y_datas, max_iterations=1000, error_threshold=0.05):
        iteration_count = 0
        while iteration_count < max_iterations:
            for x_data, y_data in six.moves.zip(x_datas, y_datas):
                self.forward(x_data)
                self.update_gradients(x_data, y_data)
            
            if self.check():
                break
            
            iteration_count += 1
    
    def forward(self, x_data):
        self.outputs = {}
        for layer, unit_weights in six.iteritems(self.weights):
            output = np.array([]).reshape(-1, 1)
            prev_output = self.outputs[layer - 1] if self.outputs else x_data
            prev_output = np.append(prev_output, 1)
            for weight in unit_weights:
                unit_sum = weight.dot(prev_output)
                activated_unit_sum = self.sigmoid(unit_sum)
                output = np.append(output, activated_unit_sum)
            self.outputs[layer] = output
        return self.outputs
    
    def update_gradients(self, x_data, y_data):
        
        
        
    
    def sigmoid(self, x):
        return 1/(1 + math.e ** (-x))
    
    
        
        