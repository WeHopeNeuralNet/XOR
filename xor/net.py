import numpy as np
import six
import math

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
                outputs = self.forward(x_data)
                self.back(x_data, y_data, outputs)
            
            if self.check():
                break
            
            iteration_count += 1
    
    def forward(self, x_data):
        outputs = {}
        for layer, unit_weights in six.iteritems(self.weights):
            output = np.array([]).reshape(-1, 1)
            prev_output = outputs[layer - 1] if outputs else x_data
            prev_output = np.append(prev_output, 1)
            for weight in unit_weights:
                unit_sum = weight.dot(prev_output)
                activated_unit_sum = self.sigmoid(unit_sum)
                output = np.append(output, activated_unit_sum)
            outputs[layer] = output
        return outputs
    
    def back(self, x_data, y_data, outputs):
        outputs[len(self.shape) - 2] = np.append(outputs[len(self.shape) - 2], 1)
        for i, last_hidden_output in enumerate(outputs[len(self.shape) - 2]):
            for j, last_output in enumerate(outputs[len(self.shape) - 1]):
                self.weights[len(self.shape) - 1][j][i] -= (y_data[j] - last_output) \
                    * last_output * (1 - last_output) * last_hidden_output
        
    def predict(self, layer):
        self.weights[layer].dot()
    
    def sigmoid(self, x):
        return 1/(1 + math.e ** (-x))
    
    
        
        