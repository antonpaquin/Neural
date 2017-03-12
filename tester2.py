from neuralnet import NeuralNet
from neuron import Neuron, Input, Output
from networkbuilder import NetworkBuilder

from random import Random
import code

nBuild = NetworkBuilder()

nBuild.addLayer(Input, 10)
nBuild.addLayer(Neuron, 10)
nBuild.addLayer(Output, 10, bias=False)

mNeurons = nBuild.get()

net = NeuralNet(mNeurons)

net.set_cost(lambda x: x * abs(x))
net.set_training_rate(1)
net.set_input_size(10)
net.set_output_size(10)

r = Random()
def generate_data():
    inp = gen_input()
    out = gen_output(inp)
    return inp, out

def gen_input():
    res = [0]*10
    res[r.randint(0, 4)] = 1
    res[r.randint(5, 9)] = 1
    return res

def gen_output(inp):
    a0 = inp[0:5].index(1)
    a1 = inp[5:10].index(1)
    res = [0]*10
    res[a0+a1] = 1
    return res

testCase = [0,0,0,1,0,0,1,0,0,0]
def do_batch(size):
    inp = []
    out = []
    for i in range(size):
        i, o = generate_data()
        inp.append(i)
        out.append(o)
    net.batch(inp, out)

    net.forward(testCase)
    print('\t'.join(list(map(lambda x: str(round(x, 3)), net.read()))))

code.interact(local=locals())
