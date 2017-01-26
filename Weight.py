import numpy as np
class Weight:

    def __init__(self, ipNodes, outNodes, mean = 0, std_dev = 0):
        self.input_nodes = ipNodes
        self.out_nodes = outNodes
        self.mean = mean
        self.std_dev = std_dev
        self.weights = np.zeros(shape= (self.input_nodes, self.out_nodes), dtype=float)
        self.initialize_weights()

    def initialize_weights(self):
        for i in range(self.input_nodes):
            self.weights[i,:] = np.random.normal(self.mean, self.std_dev, self.out_nodes)

    def update_weights(self, update):
        self.weights = self.weights + update