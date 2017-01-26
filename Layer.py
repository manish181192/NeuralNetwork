from enum import Enum
import numpy as np

class LAYER_TYPE(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class Layer:

    # size = 0 # no of perceptrons
    # ip_size = 0 # no of nodes in previous layer
    # activation_function = None # activation function of Layer
    # # nodes = None # Perceptrons
    # ip = None # ip values for the layer
    # h_activated = None # activated output of all nodes
    # h = None # list of hypothesis of all nodes
    # weight_input = None #input weight Layer
    # layer_type = None #type of layer
    # gradients = None # current set of gradients computed
    bias = 0.01
    def __init__(self, no_of_nodes, layer_type, activation_function = None, ip_weight = None):
        self.size = no_of_nodes
        self.layer_type = layer_type
        # self.nodes = []
        if layer_type!= LAYER_TYPE.INPUT:
            self.activation_function = activation_function
            self.weight_input = ip_weight
            self.ip_size = ip_weight.input_nodes
            self.gradients = np.zeros(shape= (self.ip_size, self.size))
            self.ip = np.zeros(self.ip_size)
            self.h_activated = np.zeros(self.size)
            self.h = np.zeros(self.size)
            # self.build_Layer()

    # def build_Layer(self):
    #     for i in range(self.size):
    #         current_perceptron = Perceptron(activation_function= self.activation_function)
    #         self.nodes.append(current_perceptron)

    def feed_values(self, current_input):
        if self.layer_type!=LAYER_TYPE.INPUT and len(current_input) != self.ip_size:
            print("Inconsitent input length")
            exit(0)
        self.ip = current_input

    def process_input(self):
        if self.layer_type == LAYER_TYPE.INPUT:
            self.h = self.ip
            self.h_activated = self.ip
            return "WARNING : INPUT layer cannot be processed"
        elif self.layer_type == LAYER_TYPE.HIDDEN:
            # calulate h and activation
            self.h = self.calculate_hypothesis(input_list= self.ip, weights_list=self.weight_input.weights)
            self.h_activated = self.activation_function.activate(self.h)
        elif self.layer_type == LAYER_TYPE.OUTPUT:
            # calulate softmax for each class
            self.h = self.calculate_hypothesis(input_list=self.ip, weights_list=self.weight_input.weights)
            for i in range(self.size):
                self.h_activated[i] = self.activation_function.activate(self.h, index = i)

    #   Compute gradient for each node output wrt Its Input
    def compute_gradient(self):
        for i in range(self.size):
            gredient_ = self.activation_function.get_gradient(self.ip, self.h[i], self.h_activated[i])
            self.gradients[:,i] = gredient_

    def update_weights(self):
            self.weight_input.update_weights(self.gradients)

    def get_output(self):
        return self.h

    def get_output_activated(self):
        return self.h_activated

    def calculate_hypothesis(self, weights_list, input_list):
        hypothesis_vector = np.dot(input_list, weights_list)
        hypothesis = np.add(hypothesis_vector, self.bias)
        return hypothesis

    def apply_activation(self, input_x, index= -1):
        self.output = self.activation_function.activate(input_x, index= index)
        return  self.output

    def get_result(self, weights_list, input_list, index = -1):
        if len(input_list) != len(weights_list):
            print("ERROR : weights dimension")
            exit(0)
        h = self.calculate_hypothesis(weights_list= weights_list, input_list= input_list)
        self.output = self.apply_activation(h, index)
        return self.output