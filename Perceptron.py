# from Activation_functions import Activation_Function
# import numpy as np
# # Single Perceptron
# # Parameters :  input list, activation_function, weight_list
# # Description : calulates activation_function, hypothesis
# class Perceptron:
#     input = None
#     weights = None
#     activation_function = None
#     output = None
#     bias = 0.01
#
#     def __init__(self, activation_function):
#         self.input = []
#         weights = []
#
#         self.activation_function = activation_function
#
#     def calculate_hypothesis(self, weights_list, input_list):
#         if len(input_list) != len(weights_list):
#             print("ERROR : weights dimension")
#             exit(0)
#
#         self.input = input_list
#         self.weights = weights_list
#
#         hypothesis = 0
#         # for i in range(len(self.input)):
#         #     hypothesis += (self.weights[i] * self.input[i])
#         hypothesis_vector = self.weights*self.input
#         hypothesis = np.sum(hypothesis_vector)
#         return hypothesis + self.bias
#
#     def apply_activation(self, input_x, index= -1):
#             self.output = self.activation_function.activate(input_x, index= index)
#             return  self.output
#
#     def get_output(self):
#         return self.output
#
#     def get_result(self, weights_list, input_list, index = -1):
#         if len(input_list) != len(weights_list):
#             print("ERROR : weights dimension")
#             exit(0)
#         h = self.calculate_hypothesis(weights_list= weights_list, input_list= input_list)
#         self.output = self.apply_activation(h, index)
#         return self.output