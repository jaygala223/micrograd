import math
import numpy as np
import matplotlib.pyplot as plt

# defining Value objects for storing values and their operations for backprop

# children is to know what is the heritage/parentage of a particular Value obj
# op is the previous operation which made that Value object
# label is identifier for a Value object just like variable name
# self.grad maintains the derivative of the final expression (loss fn in case of neural nets) wrt to that Value obj (weight object in nn)
# in _backward() we define how to grads will be defined at that stage / node i.e. local gradients ... note that in every def _backward() we are defining the grad 
# calculated at that node and multiplying it with the grad of the out because of the chain rule 


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = _children
        self._op = _op
        self.label = label
    
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out       
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __rmul__(self, other): # other * self
        return self * other
    
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int / float vals allowed"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other*(self.data**(other-1)) * out.grad
        
        out._backward = _backward

        return out
    
    def __neg__(self): # -self
        return self * -1 

    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __lt__(self, other): # less than
        other = other if isinstance(other, Value) else Value(other)
        return self.data < other.data

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out
    
    
    def exp(self):
        x = self.data
        e = math.exp(x)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(value):
            if value not in visited:
                visited.add(value)
                for child in value._prev:
                    build_topo(child)
                topo.append(value)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


