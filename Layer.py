from Perceptron import Perceptron
from enum import Enum
import numpy as np

class LAYER_TYPE(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class Layer:

    size = 0 # no of perceptrons
    ip_size = 0 # no of nodes in previous layer
    activation_function = None # activation function of Layer
    nodes = [] # Perceptrons
    ip = [] # ip values for the layer
    h_activated = [] # activated output of all nodes
    h = [] # list of hypothesis of all nodes
    weight_input = None #input weight Layer
    layer_type = None #type of layer
    gradients = [] # current set of gradients computed

    def __init__(self, no_of_nodes, layer_type, activation_function = None, ip_weight = None):
        self.size = no_of_nodes
        self.activation_function = activation_function
        self.weight_input = ip_weight
        self.ip_size = len(ip_weight[:,0])
        self.gradients = np.zeros(self.ip_size)

        if layer_type == LAYER_TYPE.HIDDEN:
            self.layer_type = LAYER_TYPE.HIDDEN
            self.build_Layer()
        elif layer_type == LAYER_TYPE.INPUT:
            self.layer_type = LAYER_TYPE.INPUT
        elif layer_type == LAYER_TYPE.OUTPUT:
            self.layer_type = LAYER_TYPE.OUTPUT
            self.build_Layer()

    def build_Layer(self):
        for i in range(self.size):
            current_perceptron = Perceptron(self.activation_function)
            self.nodes.append(current_perceptron)

    def feed_values(self, current_input):
        if len(current_input) != self.size:
            return "Inconsitent input length"
        self.ip = current_input

    def process_input(self):
        if self.layer_type == LAYER_TYPE.INPUT:
            self.h = self.ip
            self.h_activated = self.ip
            return "WARNING : INPUT layer cannot be processed"
        elif self.layer_type == LAYER_TYPE.HIDDEN:
            # calulate h and activation
            for i in range(self.size):
                self.h.append(self.nodes[i].calculate_hypothesis(input_list= self.ip, weights_list=self.weight_input.weights[:, i]))
                self.h_activated.append(self.nodes[i].apply_activation(self.h[i]))
        elif self.layer_type == LAYER_TYPE.OUTPUT:
            # calulate softmax for each class
            for i in range(self.size):
                self.h.append(self.nodes[i].calculate_hypothesis(input_list=self.ip, weights_list=self.weight_input.weights[:, i]))
                self.h_activated.append(self.nodes[i].apply_activation(self.h[i], index = i))

    ##  compute gradient:
    #   Compute gradient for each node output wrt Its Input
    def compute_gradient(self):
        for i in range(self.size):
            gredient_ = (self.activation_function.get_gradient(self.ip[i], self.h[i], self.h_activated[i], self.weight_input, i))
            self.gradients = self.gradients + gredient_

    def update_weights(self):
            self.weight_input.update_weights(self.gradients)

    def get_output(self):
        return self.h

    def get_output_activated(self):
        return self.h_activated