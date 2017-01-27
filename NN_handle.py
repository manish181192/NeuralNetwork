from NeuralNetwork import NeuralNetwork
import numpy as np

no_of_features = 3
train_data = [[0,1,2],[0,1,1],[1,0,1],[2,1,0]]
labels = [1,1,0,1]
hidden_layer_sizes = [10]
output_classes = 1

feed_dict = {}

feed_dict['label'] = np.asarray(labels)
feed_dict['input'] = np.asarray(train_data)

nn = NeuralNetwork(no_of_input_features= no_of_features, hidden_layer_sizes=hidden_layer_sizes, output_classes=output_classes)
# nn.set_datapoint(train_data[0], label= labels[0])
# print(nn.forward_feed_datapoint())
# nn.back_propagation_datapoint()
nn.train(no_of_epochs= 100, feed_dict = feed_dict)
