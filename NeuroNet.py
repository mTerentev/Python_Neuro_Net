import numpy as np

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
    offsets_vector=None
    weights_matrix=None
    def __init__(self, n):
        super().__init__(n)
        self.offsets_vector=np.random.random(n)
    def connect(self,previous_layer):
        self.weights_matrix=2*np.random.random((previous_layer.number_of_neurons,self.number_of_neurons))-1
    def forward_propogate(self,previous_layer_values_vector):
        self.values_vector=self.activation_function(previous_layer_values_vector.dot(self.weights_matrix)+self.offsets_vector)

net=Net(
    InitLayer(3),
    HiddenLayer(5),
    HiddenLayer(5),
    HiddenLayer(5)
)
print(net.forward_propogate([1,2,3]))