from NeuralNetwork import NeuralNetwork

no_of_features = 3
train_data = [[0,1,2],[0,1,1],[1,0,1],[2,1,0]]
labels = [12,9,8,16]
hidden_layer_sizes = [10]
output_classes = 1

nn = NeuralNetwork(no_of_input_features= no_of_features, hidden_layer_sizes=hidden_layer_sizes, output_classes=output_classes)
nn.set_datapoint(train_data[0])
nn.forward_feed_datapoint()
nn.back_propagation_datapoint()