from Layer import Layer,LAYER_TYPE
from Weight import Weight
from Activation_functions import ACTIVATION_FUNCTION,Activation_Function

#Neural Network
class NeuralNetwork:

    current_input = []

    #### ARCHITECTURE ##
    no_of_input_features = None
    # no of elements = no of hidden layers
    # hidden_layer_sizes[i] = no of nodes in hidden layer 'i'
    hidden_layer_sizes = []
    output_classes = None

    input_layer = None
    W1 = None
    h1 = None
    W2 = None
    output_layer = None

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

    def set_datapoint(self, input_datapoint):
        if len(input_datapoint) != self.no_of_input_features:
            return None
        self.current_input = input_datapoint

    #Perform forward feed
    def forward_feed_datapoint(self):

        self.input_layer.feed_values(self.current_input)
        self.h1.feed_values(self.current_input)
        self.h1.process_input()
        hidden_layer_out_1 = self.h1.get_output()
        self.output_layer.feed_values(hidden_layer_out_1)
        self.output_layer.process_input()
        prob_dist = self.output_layer.get_output()
        print("Probability Distribution"+str(prob_dist))

    #Perform backpropagation
    def back_propagation_datapoint(self):
        # compute gradient for output
        self.output_layer.compute_gradient()
        self.output_layer.update_weights()
        #compute gradient for hidden layer
        self.h1.compute_gradient()
        self.h1.update_weights()