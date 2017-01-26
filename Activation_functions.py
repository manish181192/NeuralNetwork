import math as m
from enum import Enum
import numpy as np
class ACTIVATION_FUNCTION(Enum):
    SIGMOID = 0
    SOFTMAX = 1
    RELU = 2
    TANH = 3
class Activation_Function:

    #default sigmoid
    id = ACTIVATION_FUNCTION.SIGMOID
    exp_list = None
    sum_exponentials = None

    def __init__(self, id):
        self.id = id

    def activate(self, ip, index = -1):
        if self.id == ACTIVATION_FUNCTION.SIGMOID:
            return self.sigmoid(ip)
        elif self.id == ACTIVATION_FUNCTION.SOFTMAX:
            # softmax
            if self.exp_list == None:
                self.calc_exp_list(ip)
            return self.softmax(index)

    # SIGMOID ACTIVATION FUNCTION
    def sigmoid(self, ip):
        if ip == 0:
            return 1
        return 1 / (1 + m.exp(-1 * ip))

    def calc_exp_list(self, ip):
        self.exp_list = np.exp(ip)
        self.sum_exponentials = np.sum(self.exp_list)
        print("Softmax : "+self.exp_list)

    #SOFTMAX ACTIVATION FUNCTION
    def softmax(self, k):
        # ip : hypothesis of all classes
        return self.exp_list[k]/self.sum_exponentials

    ##### GRADIENT CALCULATOR #####
    def get_gradient(self, ip, out, out_activated):
        if self.id == ACTIVATION_FUNCTION.SIGMOID:
            self.sigmoid_gradient(ip, out, out_activated)
        if self.id == ACTIVATION_FUNCTION.SOFTMAX:
            self.softmax_gradient(ip, out, out_activated)

    def sigmoid_gradient(self, ip, out, out_activated, weight, index):
        one = np.ones(len(out_activated))
        one_minus_sigmoid = one - out_activated
        sig_mul_one_minus_sig = out_activated*one_minus_sigmoid

        gradient = ip * sig_mul_one_minus_sig

        return gradient

    def softmax_gradient(self, ip, out, out_activated):
        sum_exponentials = self.sum_exponentials
        coeff = -1*out/sum_exponentials

        return 0