from neuron import Neuron, Bias, Input, Output, connect
from code import interact

## Bulk operators

class NeuralNet:
    def __init__(self, neurons):
        self.network = neurons
        self.costf = lambda x: 0

        self.input_size = 0
        self.output_size = 0


    # Metaparameters

    def set_cost(self, costf):
        self.costf = costf

    def set_training_rate(self, rate):
        for neuron in self.network:
            neuron.training_rate = rate

    def set_input_size(self, size):
        self.input_size = size

    def set_output_size(self, size):
        self.output_size = size


    # Network operations

    def forward(self, inputs):
        inputLayer = self.network[:self.input_size]
        for neuron, value in zip(inputLayer, inputs):
            neuron.forward(value)
        for neuron in self.network[self.input_size:]:
            neuron.forward()

    def backward(self, expected):
        rNetwork = list(reversed(self.network))
        outputLayer = rNetwork[:self.output_size]
        outputs = list(reversed(expected))
        for neuron, value in zip(outputLayer, outputs):
            neuron.backward(self.costf, value)
        for neuron in rNetwork[self.output_size:]:
            neuron.backward()

    def apply_deltas(self):
        for neuron in self.network:
            neuron.apply_deltas()

    def read(self):
        outputs = self.network[len(self.network) - self.output_size :]
        res = []
        for neuron in outputs:
            res.append(neuron.activation)
        return res


    # training!

    def batch(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            self.forward(inp)
            self.backward(out)
        self.apply_deltas()
