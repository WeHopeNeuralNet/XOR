from enum import Enum
import math

class LayerType(Enum):
    input_layer = 1
    hidden_layer = 2
    output_layer = 3
    
class Unit(object):
    
    def __init__(self, layertype = LayerType.input_layer):
        self.layertype = layertype
        self.weights = None
        
    def activate(self, input_values):
        if self.weights == None:
            self.init_weights()
        u = self.weights.dot(input_values)
        
        # TODO Must add other activation function.
        return self.sigmoid(u)
    
    def update(self, update_value):
        self.weights += update_value
        
    def relu(self, u):
        return max(u, 0)
    
    def sigmoid(self, u):
        return 1 / (1 + math.exp(-u))
        
    
        