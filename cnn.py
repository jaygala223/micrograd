from engine import Value
from mlp import MLP, Layer, Neuron
import numpy as np

def flatten(image, image_size):
    height = image_size
    width = image_size

    flat_array = []

    flat_array = [0]*(height*width)
    cnt = 0

    for i in range(height):
        for j in range(width):
            flat_array[cnt] = image[i][j]
            cnt += 1

    return flat_array

def mse_loss(ypred, y_train):
    return (ypred - y_train)**2

class MaxPool2d:
    def __init__(self, image_size, filter_size):
        self.image_size = image_size
        self.filter_size = filter_size
    
    def __call__(self, image):
        max_items = []
        final_image_size = self.image_size // 2
        final_image =  [[Value(0.0) for x in range(final_image_size)] for y in range(final_image_size)]
        
        for i in range(0, self.image_size, self.filter_size):
            for j in range(0, self.image_size, self.filter_size):
                max_item = Value(-1e9)
                for x in range(self.filter_size):
                    for y in range(self.filter_size):
                        max_item = max(Value(image[i+x][j+y]), max_item)
                max_items.append(max_item)
        
        idx = 0
        for a in range(final_image_size):
            for b in range(final_image_size):
                final_image[a][b] += max_items[idx]
                idx += 1        
        return final_image

class Conv2d:
    def __init__(self, image_size, filter_size):
        self.image_size = image_size
        self.filter_size = filter_size
        self.filter = [[Value(5.0, label='conv') for _ in range(filter_size)] for _ in range(filter_size)]

    def __call__(self, image):
        final_image_size = int(self.image_size - self.filter_size + 1)
        # print(final_image_size, type(final_image_size))
        final_image = np.zeros([final_image_size, final_image_size])
        final_image = [[Value(0.0) for x in range(final_image_size)] for y in range(final_image_size)] 
        
        for i in range(final_image_size):
            for j in range(final_image_size):
                
                for x in range(self.filter_size):
                    for y in range(self.filter_size):

                        if i+x < self.image_size and j+y < self.image_size:
                            final_image[i][j] += (image[i+x][j+y] * self.filter[x][y])

        return final_image
    
    def parameters(self):
        o = flatten(self.filter, self.filter_size)
        return o

import torch.nn as nn
import torch.nn.functional as F


class CNN:
    def __init__(self):
        self.conv1 = Conv2d(5, 2)
        self.max_pool = MaxPool2d(4,2)
        self.fc = MLP(4, [6,6,1])
    
    def __call__(self, input_image):
        outputs = self.conv1(input_image)
        outputs = self.max_pool(outputs)
        # print(outputs)
        outputs = flatten(outputs, 2)

        t = []
        for i in outputs:
            t.append(i.data)
        
        outputs = self.fc(t)
        return outputs
    
    def parameters(self):
        # print(self.conv1.parameters())
        return self.fc.parameters() + self.conv1.parameters()
    
if __name__ == "__main__":

    image = [[4,5,4,4,4],
            [4,5,4,4,4],
            [4,5,4,4,4],
            [4,5,4,4,4],
            [4,5,4,4,4]]

    label = 1.0

    cnn_model = CNN()

    for i in range(5):
        
        #forward
        outputs = cnn_model(image)
        loss = mse_loss(outputs, label)

        # zero_grad
        for p in cnn_model.parameters():
            p.grad = 0.0

        loss.backward()

        # update
        for p in cnn_model.parameters():
            p.data += -0.1 * p.grad

        print(f"Epoch {i}, Loss: {loss.data}")