import math

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError as e:
        return 0

def sigmoid_p(x):
    return x * (1-x)
