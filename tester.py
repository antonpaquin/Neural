from neuron import Neuron, Bias, Input, Output, connect
from random import Random
import code


r = Random()

## Create layers

layer1 = []
layer1Size = 10
for i in range(layer1Size):
    n = Input()
    n.training_rate = 0.05
    layer1.append(n)

layer2 = []
layer2Size = 10
for i in range(layer2Size):
    n = Neuron()
    n.training_rate = 0.05
    layer2.append(n)

layer3 = []
layer3Size = 10
for i in range(layer3Size):
    n = Output()
    n.training_rate = 0.05
    layer3.append(n)


## Connect nodes

for front in layer1:
    for back in layer2:
        connect(front, back)

for front in layer2:
    for back in layer3:
        connect(front, back)


## Bulk operators

def forward(inputs):
    for n, val in zip(layer1, inputs):
        n.forward(val)
    for n in layer2:
        n.forward()
    for n in layer3:
        n.forward()

def backward(costf, expected):
    for n, val in zip(layer3, expected):
        n.backward(costf, val)
    for n in layer2:
        n.backward()
    for n in layer1:
        n.backward()

def apply_deltas():
    for n in layer1:
        n.apply_deltas()
    for n in layer2:
        n.apply_deltas()
    for n in layer3:
        n.apply_deltas()


## Full meta

def generate_example():
    res = [0]*10
    res[r.randint(0,9)] = 1
    return res, res

batch_size = 1000

def batch():
    costf = lambda x: x * abs(x)

    for i in range(batch_size):
        inp, out = generate_example()
        forward(inp)
        backward(costf, out)
    apply_deltas()

def test(inp):
    forward(inp)
    out = []
    for n in layer3:
        out.append(n.read())
    return out

testCase = [0,0,0,1,0,0,0,0,0,0]

def train():
    while(True):
        for i in range(10):
            batch()
        n = list(map(lambda x: str(round(x, 3)), test(testCase)))
        print('\t'.join(n))

code.interact(local=locals())
