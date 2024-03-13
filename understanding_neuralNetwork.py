import numpy as np
import matplotlib as mp

input_vector = np.array([2, 1.5])
weight_1 = np.array([1.45, -0.66])
bias = np.array([0.0])

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2

prediction = make_prediction(input_vector, weight_1, bias)

print (f"the prediction result is {prediction}")