# NeuralNet-Python
Implementation for a simple neural network in Python
  
Dependencies:
  * numpy (for computations)
  * pickle (for serializing and deserializing weights if needed)
  * random (for initializing weights)
  
Some examples are given in the main method of the NeuralNetwork.py file.
  
The learning rate, momentum and number of epochs can be edited. Two important methods:
  * exhaustive_train() -> trains the neural net
  * estimate() -> estimates the output layer based on an input layer given as argument
  
Example for XOR prediction:
```python
    network = RatesNeuralNetwork([[0, 0], [1, 1], [1, 0], [0, 1]], [[0], [0], [1], [1]], 0)
    network.LEARNING_RATE = 1
    network.MOMENTUM = 0
    network.EPOCHS = 16000
    network.exhaustive_train()
    print("%.4f" % network.estimate([0, 0])[0])
    print("%.4f" % network.estimate([1, 1])[0])
    print("%.4f" % network.estimate([1, 0])[0])
    print("%.4f" % network.estimate([0, 1])[0])
```
