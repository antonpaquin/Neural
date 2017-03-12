from neuron import Bias, connect

class NetworkBuilder:
    def __init__(self):
        self.network = []
        self.layers = []

    def get(self):
        return self.network

    def addLayer(self, layerType, size, bias=True):
        thisLayer = []
        for i in range(size):
            n = layerType()
            thisLayer.append(n)
            self.network.append(n)

        if len(self.layers) != 0:
            for front in self.layers[-1]:
                for back in thisLayer:
                    connect(front, back)

        if bias:
            n = Bias()
            thisLayer.append(n)
            self.network.append(n)

        self.layers.append(thisLayer)
