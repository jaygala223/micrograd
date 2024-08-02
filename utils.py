import math
import numpy as np
from engine import Value

def softmax(logits):
    outputs = []

    denominator = Value(0)

    for logit in logits:
        denominator += logit.exp()

    for logit in logits:
        softmax_score = logit.exp() / denominator
        outputs.append(softmax_score)

    return outputs

# usage: softmax([Value(data=0.8942415841754484), 
#          Value(data=-0.7502373687945066), 
#          Value(data=0.32914364144553077)]
# )


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


def cross_entropy_loss(logits, true_labels):
    """
    logits and true_labels both should be list of Value objects ... 
    assuming logits are prob values from softmax
    
    returns cross_entropy_loss as a Value object
    """
    loss = Value(0)

    for i in range(len(logits)):
        loss += -1 * math.log(logits[i]) * true_labels[i]

    return loss

# usage: cross_entropy_loss(logits=[0.1,0.2,0.7], true_labels=[0,0,1])

from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot