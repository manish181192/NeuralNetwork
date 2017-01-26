from Layer import Layer,LAYER_TYPE
from Weight import Weight
from Activation_functions import ACTIVATION_FUNCTION,Activation_Function
import numpy as np
#Neural Network
class NeuralNetwork:

    # current_input = []
    # current_label =[]
    # #### ARCHITECTURE ##
    # no_of_input_features = None
    # # no of elements = no of hidden layers
    # # hidden_layer_sizes[i] = no of nodes in hidden layer 'i'
    # hidden_layer_sizes = []
    # output_classes = None
    #
    # input_layer = None
    # W1 = None
    # h1 = None
    # W2 = None
    # output_layer = None

    def __init__(self ,no_of_input_features, hidden_layer_sizes, output_classes):

        self.no_of_input_features = no_of_input_features
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_classes = output_classes

        self.input_layer = Layer(no_of_nodes= no_of_input_features, layer_type= LAYER_TYPE.INPUT)
        self.W1 = Weight(ipNodes= self.no_of_input_features, outNodes= self.hidden_layer_sizes[0], std_dev= 0.1)
        self.h1 = Layer(no_of_nodes= self.hidden_layer_sizes[0],
                        layer_type= LAYER_TYPE.HIDDEN,
                        ip_weight= self.W1,
                        activation_function= Activation_Function(ACTIVATION_FUNCTION.SIGMOID))
        self.W2 = Weight(ipNodes=self.hidden_layer_sizes[0], outNodes=self.output_classes, std_dev=0.1)
        self.output_layer = Layer(no_of_nodes= self.output_classes,
                                  layer_type= LAYER_TYPE.OUTPUT,
                                  ip_weight= self.W2,
                                  activation_function= Activation_Function(ACTIVATION_FUNCTION.SOFTMAX))

    def set_datapoint(self, input_datapoint, label):
        if len(input_datapoint) != self.no_of_input_features:
            return None
        self.current_input = input_datapoint
        self.current_label = label

    #Perform forward feed
    def forward_feed_datapoint(self):

        self.input_layer.feed_values(self.current_input)
        self.h1.feed_values(self.current_input)
        self.h1.process_input()
        hidden_layer_out_1 = self.h1.get_output_activated()
        self.output_layer.feed_values(hidden_layer_out_1)
        self.output_layer.process_input()
        prob_dist = self.output_layer.get_output_activated()
        # print("Probability Distribution"+str(prob_dist))
        # -log(prediction) label = 1
        # -log(1-prediction) label =0
        total_loss = 0
        for i in range(self.output_classes):
            if self.current_label == 0:
                total_loss = total_loss + np.log(1-prob_dist)
            elif self.current_label == 1:
                total_loss = total_loss + np.log(prob_dist)
        return -1*total_loss

    #Perform backpropagation
    def back_propagation_datapoint(self):
        # compute gradient for output
        self.output_layer.compute_gradient()
        self.output_layer.update_weights()
        #compute gradient for hidden layer
        self.h1.compute_gradient()
        self.h1.update_weights()

    def train(self, no_of_epochs, feed_dict = None):
        self.no_of_training_data = len(feed_dict['label'])
        total_cost = np.zeros(self.no_of_training_data)
        for i in range(no_of_epochs):
            for j in range(self.no_of_training_data):
                self.set_datapoint(feed_dict['input'][j], feed_dict['label'][j])
                total_cost[j] =self.forward_feed_datapoint()
                self.back_propagation_datapoint()
            print("EPOCH "+ str(i) +" : \n Total Loss :" + str(np.sum(total_cost)))