from .helper import *
from random import Random

class Neuron:
    def __init__(self):
        self._forward_connections = []

        self._backward_connections = []

        self.backward_weights = {}

        self.activation = 0

        self._backward_deltas = {}
        self.erf = 0
        self.training_rate = 0
        self.batch_size = 0

    def forward(self):
        s = 0
        for connection in self._backward_connections:
            weight = self.backward_weights[connection]
            s += (connection.activation * weight)

        self.activation = sigmoid(s)

    def backward(self):
        s = 0
        for connection in self._forward_connections:
            weight = connection.backward_weights[self]
            s += weight * connection.erf
        s *= sigmoid_p(self.activation)

        self.erf = s

        for connection in self._backward_connections:
            self._backward_deltas[connection] += connection.activation * self.erf

        self.batch_size += 1

    def apply_deltas(self):
        if self.batch_size == 0:
            return

        for connection in self._backward_connections:
            delta = self._backward_deltas[connection] * self.training_rate / self.batch_size
            self.backward_weights[connection] += delta
            self._backward_deltas[connection] = 0

        self.batch_size = 0

class Bias:
    def __init__(self):
        self._forward_connections = []
        self.activation = 1

    def forward(self):
        pass

    def backward(self):
        pass

    def apply_deltas(self):
        pass

class Input:
    def __init__(self):
        self._forward_connections = []
        self.activation = 0

    def forward(self, value):
        self.activation = value

    def backward(self):
        pass

    def apply_deltas(self):
        pass

class Output:
    def __init__(self):
        self._backward_connections = []
        self.backward_weights = {}

        self.activation = 0

        self._backward_deltas = {}
        self.erf = 0
        self.training_rate = 0
        self.batch_size = 0

    def forward(self):
        s = 0
        for connection in self._backward_connections:
            weight = self.backward_weights[connection]
            s += (connection.activation * weight)

        self.activation = sigmoid(s)

    def backward(self, costf, expected):
        self.erf = costf(expected - self.activation)

        for connection in self._backward_connections:
            self._backward_deltas[connection] += connection.activation * self.erf

        self.batch_size += 1

    def apply_deltas(self):
        if self.batch_size == 0:
            return

        for connection in self._backward_connections:
            delta = self._backward_deltas[connection] * self.training_rate / self.batch_size
            self.backward_weights[connection] += delta
            self._backward_deltas[connection] = 0

        self.batch_size = 0

    def read(self):
        return self.activation

r = Random()

def connect(back, front):
    back._forward_connections.append(front)
    front._backward_connections.append(back)
    front.backward_weights[back] = 2*r.random()-1
    front._backward_deltas[back] = 0
