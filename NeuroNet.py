import numpy as np
from numpy import random

class Net():
    layers=[]
    def __init__(self,*args):
        self.layers=args
        previous_layer=None
        for layer in self.layers:
            if type(layer)!=InitLayer:
                layer.connect(previous_layer)
            previous_layer=layer
    def forward_propogate(self,input):
        self.layers[0].set_input(input)
        for layer in self.layers:
            if type(layer)!=InitLayer:
                layer.forward_propogate(previous_layer.values_vector)
            previous_layer=layer
        return previous_layer.values_vector
    
    def backward_propogate(self,target,i=1):
        layer=self.layers[i]
        previous_layer=self.layers[i-1]
        layer.forward_propogate(previous_layer.values_vector,True)
        if i==len(self.layers)-1:
            layer.delta_l=(layer.values_vector-np.array(target))*layer.activation_function_derivative
            return 0
        self.backward_propogate(target,i+1)
        next_layer=self.layers[i+1]
        layer.delta_l=layer.activation_function_derivative*(next_layer.delta_l.dot(np.transpose(next_layer.weights_matrix)))

    def train(self,input,target):
        self.layers[0].set_input(input)
        self.backward_propogate(target)
        for layer in self.layers:
            if type(layer)!=InitLayer:
                layer.weights_update(0.01)
    def print(self):
        for layer in self.layers:
            print(layer.values_vector)
class Layer():
    number_of_neurons=0
    values_vector=None
    def __init__(self,n):
        self.number_of_neurons=n
        self.values_vector=np.zeros(n)

class InitLayer(Layer):
    def set_input(self,i):
        self.values_vector=np.array(i)

class HiddenLayer(Layer):
    activation_function=lambda self,x: 1/(1+np.exp(-x))
    activation_function_derivative=None
    delta_l=None
    previous_layer_values_vector=None
    offsets_vector=None
    weights_matrix=None
    def __init__(self, n):
        super().__init__(n)
        self.offsets_vector=np.random.random(n)
    def connect(self,previous_layer):
        self.weights_matrix=2*np.random.random((previous_layer.number_of_neurons,self.number_of_neurons))-1
    def forward_propogate(self,previous_layer_values_vector,learning=False):
        self.values_vector=self.activation_function(previous_layer_values_vector.dot(self.weights_matrix)+self.offsets_vector)
        if learning:
            self.activation_function_derivative=self.values_vector*(1-self.values_vector)
            self.previous_layer_values_vector=previous_layer_values_vector
    def weights_update(self,a):
        self.weights_matrix-=a*(np.transpose(self.delta_l).dot(self.previous_layer_values_vector))
net=Net(
    InitLayer(1),
    HiddenLayer(5),
    HiddenLayer(5),
    HiddenLayer(1)
)
# for i in range(100):
#     net.train(random.random(),1)
#     print(net.forward_propogate(random.random()))
net.print()
net.layers[0].set_input(5)
pre=net.layers[0]
for layer in net.layers[1:]:
    layer.forward_propogate(pre.values_vector,True)
    pre=layer
net.layers[3].delta_l=np.array([[1,2,3,4,5]])
net.layers[3].weights_update(0.5)
net.print()
