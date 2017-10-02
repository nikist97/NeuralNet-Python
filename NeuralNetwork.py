"""
This module implements a simple, yet, powerful neural network.

For the implementation of the neural network I use a Multilayer Perceptron.
The activation function is the following sigmoid function - f(x) = 1/(1 + exp(-x))

This implementation uses 3 layers: 1 Input Layer, 1 Hidden Layer (4 neurons) and 1 Output Layer
"""

import numpy
import pickle
import random


class RatesNeuralNetwork(object):
    """
    A utility class which builds a neural network
    """

    LEARNING_RATE = 1
    MOMENTUM = 0
    EPOCHS = 20000

    # noinspection PyUnusedLocal
    def __init__(self, inputs, outputs, threshold):
        """
        constructor sets the input values and the threshold to use
        :param inputs: the values for the input layer of the neural net
        (should be 2D list with all the training input data)
        :param outputs: the values for the output layer of the neural net
        (should be 2D list with all the training output data)
        :param threshold: the threshold used for the neuron value computation
        """

        self.input_layer_data = numpy.array(inputs)
        self.output_layer_data = numpy.array(outputs)

        assert len(self.output_layer_data) == len(self.input_layer_data)

        self.hidden_layers = [[Neuron(connections_count=len(self.input_layer_data[0]), fake=(i == 4), value=threshold)
                               for i in range(5)], ]
        self.output_layer = [Neuron(connections_count=len(self.hidden_layers[-1]), fake=False)
                             for i in range(len(self.output_layer_data[0]))]

        self.layers_gradients = []
        for i in range(len(self.hidden_layers)):
            self.layers_gradients.append([0] * len(self.hidden_layers[i]))
        self.layers_gradients.append([0] * len(self.output_layer))

    def get_current_error(self):
        """
        this method calculates the error given the current weights
        :return: the current estimated error
        """
        current_output_layer_errors = [0] * len(self.output_layer)
        current_training_set_errors = []
        index = 0
        while index < len(self.input_layer_data):
            # get the input layer and the expected output layer from the training data set
            input_layer = self.input_layer_data[index]
            real_output_layer = self.output_layer_data[index]
            assert len(real_output_layer) == len(self.output_layer)

            # compute the hidden layers values by using the current weights in the network
            last_layer_data = input_layer
            for hidden_layer in self.hidden_layers:
                for neuron in hidden_layer:
                    neuron.compute_value(input_values=last_layer_data)
                last_layer_data = numpy.array([neuron.value for neuron in hidden_layer])

            # compute the output layer values based on the current weights in the network
            for neuron in self.output_layer:
                neuron.compute_value(input_values=last_layer_data)

            # compute the current errors
            for neuron_index, neuron in enumerate(self.output_layer):
                current_output_layer_errors[neuron_index] = neuron.compute_value_error(real_output_layer[neuron_index])

            current_training_set_errors.append(sum(current_output_layer_errors))

            index += 1

        return sum(current_training_set_errors), sum(current_training_set_errors) / len(current_training_set_errors), len(current_training_set_errors)

    def train(self):
        """
        this method starts the training of the data using a backpropagation algorithm
        """

        index = 0
        while index < len(self.input_layer_data):
            # get the input layer and the expected output layer from the training data set
            input_layer = self.input_layer_data[index]
            real_output_layer = self.output_layer_data[index]
            assert len(real_output_layer) == len(self.output_layer)

            # compute the hidden layers values by using the current weights in the network
            last_layer_data = input_layer
            for hidden_layer in self.hidden_layers:
                for neuron in hidden_layer:
                    neuron.compute_value(input_values=last_layer_data)
                last_layer_data = numpy.array([neuron.value for neuron in hidden_layer])

            # compute the output layer values based on the current weights in the network
            for neuron in self.output_layer:
                neuron.compute_value(input_values=last_layer_data)

            # calculate the output gradient
            for neuron_index, neuron in enumerate(self.output_layer):
                self.layers_gradients[-1][neuron_index] = \
                    neuron.compute_value_gradient(real_output_layer[neuron_index])

            # backpropagation to calculate the gradient for the weights in the hidden layers
            for reverse_layer_index, hidden_layer in enumerate(self.hidden_layers[:: -1]):
                layer_index = len(self.hidden_layers) - 1 - reverse_layer_index
                for neuron_index, neuron in enumerate(hidden_layer):
                    following_layer_gradients = numpy.array(self.layers_gradients[layer_index + 1])
                    if layer_index == len(self.hidden_layers) - 1:
                        following_layer_weights = numpy.array([next_neuron.weights[neuron_index]
                                                               for next_neuron in self.output_layer])
                    else:
                        following_layer_weights = numpy.array([next_neuron.weights[neuron_index]
                                                               for next_neuron in self.hidden_layers[layer_index + 1]])

                    self.layers_gradients[layer_index][neuron_index] = \
                        RatesNeuralNetwork.sigmoid_derivative(neuron.value) * numpy.dot(following_layer_gradients,
                                                                                        following_layer_weights)

            # update the weights in the hidden layers with the calculated gradients
            for layer_index, hidden_layer in enumerate(self.hidden_layers):
                for neuron_index, neuron in enumerate(hidden_layer):
                    delta_weight = 0
                    for weight_index in range(len(neuron.weights)):
                        if (layer_index - 1) >= 0:
                            value = self.hidden_layers[layer_index - 1][weight_index].value
                        else:
                            value = input_layer[weight_index]
                        learn_delta = self.LEARNING_RATE * self.layers_gradients[layer_index][
                            neuron_index] * value
                        momentum_delta = self.MOMENTUM * delta_weight
                        neuron.weights[weight_index] += learn_delta + momentum_delta
                        delta_weight = learn_delta

            # update the wights in the output layer
            for neuron_index, neuron in enumerate(self.output_layer):
                delta_weight = 0
                for weight_index in range(len(neuron.weights)):
                    learn_delta = self.LEARNING_RATE * self.layers_gradients[-1][neuron_index] * \
                                  self.hidden_layers[-1][weight_index].value
                    momentum_delta = self.MOMENTUM * delta_weight
                    neuron.weights[weight_index] += learn_delta + momentum_delta
                    delta_weight = learn_delta

            # proceed with the next training data
            index += 1

    def exhaustive_train(self):
        total_error, average_error, data_set_size = self.get_current_error()

        print("Initial total error is %.6f for data set with length %d, Average error is %.6f"
              % (total_error, data_set_size, average_error))

        for i in range(self.EPOCHS):
            self.train()

            total_error, average_error, data_set_size = self.get_current_error()
            print("Current total error is %.6f for data set with length %d, Average error is %.6f, Current epoch is %d"
                  % (total_error, data_set_size, average_error, i + 1))

        total_error, average_error, data_set_size = self.get_current_error()
        print("Total error after training is %.6f for data set with length %d, Average error is %.6f"
              % (total_error, data_set_size, average_error))

    def estimate(self, input_data):
        """
        this method takes as an argument the input data and gives the output values using the trained weights
        :param input_data: the input data set  of attributes
        :return: the output data set
        """
        # compute the hidden layers values by using the current weights in the network
        last_layer_data = numpy.array(input_data)
        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                neuron.compute_value(input_values=last_layer_data)
            last_layer_data = numpy.array([neuron.value for neuron in hidden_layer])

        # compute the output layer values based on the current weights in the network
        for neuron in self.output_layer:
            neuron.compute_value(input_values=last_layer_data)

        return [neuron.value for neuron in self.output_layer]

    def get_weights(self):
        """
        this method gets all weights in the neural network in the row of the layers
        :return: a tuple containing all weights
        """

        weights = ()
        for layer in self.hidden_layers:
            for neuron in layer:
                for weight in neuron.weights:
                    weights += (weight,)
        for neuron in self.output_layer:
            for weight in neuron.weights:
                weights += (weight,)

        return weights

    def serialize_weights(self, path="weights.dat"):
        """
        a method used to serialize the weights of the neural network so that the network doesn't need
        training every time it starts
        :param path: the path of the file where the weights are serialized
        """

        weights = self.get_weights()

        with open(path, "wb") as file:
            pickle.dump(weights, file)

    def deserialize_weights(self, path="weights.dat"):
        """
        deserializes weights saved in a file
        :param path: the path of the file, which contains the weights
        """

        with open(path, "rb") as file:
            weights = tuple(pickle.load(file))

        current_weight_index = 0
        for layer in self.hidden_layers:
            for neuron in layer:
                for index in range(len(neuron.weights)):
                    neuron.weights[index] = weights[current_weight_index]
                    current_weight_index += 1
        for neuron in self.output_layer:
            for index in range(len(neuron.weights)):
                neuron.weights[index] = weights[current_weight_index]
                current_weight_index += 1

    @staticmethod
    def sigmoid(x):
        """
        the activation function which is used for the neural network computations
        """

        return 1 / (1 + numpy.exp(-1 * x))

    @staticmethod
    def sigmoid_derivative(x):
        """
        :param x: an output from the sigmoid function
        :return the value of the derivative of the sigmoid function assuming that 'x'
        is an output from the sigmoid function
        """

        return x * (1 - x)


class Neuron(object):
    """
    This class represents a single neuron in the network by encapsulating its state
    """

    def __init__(self, connections_count, fake, value=None):
        """
        constructs the neuron with None value and random weights between -0.5 and 0.5
        :param connections_count: this is the number of neurons in the previous layer
        :param fake: True if the neuron is a BIAS neuron and false otherwise
        :param value: the value set in the constructor if this is a BIAS neuron
        """

        self.value = value
        self.fake = fake
        if self.fake:
            assert self.value is not None
        # noinspection PyUnusedLocal
        self.weights = numpy.array([random.uniform(-0.5, 0.5) for i in range(connections_count)])

    def compute_value(self, input_values):
        """
        this method computes the value of the neuron, given the input data from the previous layer
        :param input_values: the input values from the neurons in the previous layer (numpy 1D array)
        """
        if self.fake:
            return

        assert len(input_values) == len(self.weights)
        self.value = RatesNeuralNetwork.sigmoid(numpy.dot(input_values, self.weights))

    def compute_value_error(self, real_value):
        """
        this method computes the error in the value by taking into account the real value of the neuron
        :param real_value: the real expected value that the network should compute
        :return: the calculated error between the two outputs
        """

        assert self.value is not None
        return 0.5 * ((real_value - self.value) ** 2)

    def compute_value_gradient(self, real_value):
        """
        this method computes the gradient for the neuron - used for output layer neurons
        :param real_value: the real expected value that the network should compute
        :return: the calculated gradient for this neuron (how much the calculated weight value affects the error)
        """

        return (real_value - self.value) * RatesNeuralNetwork.sigmoid(self.value)


if __name__ == "__main__":
    """
    some examples to test the behaviour of the neural network
    """
    # network = RatesNeuralNetwork([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]], [[0], [1], [1], [0]], 0)
    # network.exhaustive_train()
    # print("%.4f" % network.estimate([0, 0, 1])[0])
    # print("%.4f" % network.estimate([1, 1, 1])[0])
    # print("%.4f" % network.estimate([1, 0, 1])[0])
    # print("%.4f" % network.estimate([0, 1, 1])[0])
    network = RatesNeuralNetwork([[0, 0], [1, 1], [1, 0], [0, 1]], [[0], [0], [1], [1]], 0)
    network.exhaustive_train()
    print("%.4f" % network.estimate([0, 0])[0])
    print("%.4f" % network.estimate([1, 1])[0])
    print("%.4f" % network.estimate([1, 0])[0])
    print("%.4f" % network.estimate([0, 1])[0])
