from engine import Value
import numpy
import random

class Neuron:
    def __init__(self, num_inputs):
        # initialize random weights and biase for the neuron
        self.weights = [Value(random.uniform(-1,1)) for _ in range(num_inputs)]
        self.biases = Value(random.uniform(-1,1))

    def __call__(self, x):

        
        output = sum((wi*xi for wi, xi in zip(self.weights, x)), self.biases)
        output = output.tanh()
        return output
    
    def parameters(self):
        return self.weights + [self.biases]
    
class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self, x):
        output = [n(x) for n in self.neurons]
        return output[0] if len(output) == 1 else output
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        # params = []
        # for neuron in self.neurons:
        #     params.extend(neuron.parameters())
        # return params
    
class MLP:
    def __init__(self, num_inputs, num_outputs: list):
        size = [num_inputs] + num_outputs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(size)-1)] # as i+1 will go out of index

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) # output of layer i is input of layer i+1
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

x = [2.0, 3.0]
model = MLP(3, [3,4,4,1])

x_train = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

y_train = [1.0, -1.0, -1.0, 1.0]

# ypred = [model(x) for x in x_train]
# ypred

def mse_loss(ypred, y_train):
    return (ypred - y_train)**2

for k in range(200):

    #forward
    ypreds = [model(x) for x in x_train]
    loss = sum(mse_loss(y_hat, y_actual) for y_hat, y_actual in zip(ypreds, y_train))

    # zero_grad
    for p in model.parameters():
        p.grad = 0.0

    loss.backward()

    # update
    for p in model.parameters():
        p.data += -0.01 * p.grad

    if k%20 == 0: 
        print(f"Epoch {k}, Loss: {loss.data}")