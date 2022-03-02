import cupy as cp
import matplotlib.pyplot as plt
class Layer():
    values_vector=None
    previous_layer=None
    def __init__(self,n):
        self.values_vector=cp.zeros((n,1))
    def connect(self,previous_layer):
        self.previous_layer=previous_layer

class InputLayer(Layer):
    def __init__(self, n):
        super().__init__(n)
    def set_icput(self,i):
        self.values_vector=cp.transpose(cp.array([i]))

class OutLayer(Layer):
    def __init__(self):
        super().__init__(0)
    def output(self):
        return self.previous_layer.values_vector

class HiddenLayer(Layer):
    activation_function=lambda self,x: 1/(1+cp.exp(-x))
    activation_function_derivative=None
    delta_l=None
    offsets_vector=None
    weights_matrix=None

    def connect(self,previous_layer):
        super().connect(previous_layer)
        self.weights_matrix=2*cp.random.random((len(self.values_vector),len(previous_layer.values_vector)))-1

    def __init__(self, n):
        super().__init__(n)
        self.offsets_vector=cp.zeros((n,1))
    
    def forward_propogate(self,learning=False):
        self.values_vector=self.activation_function(self.weights_matrix.dot(self.previous_layer.values_vector)+self.offsets_vector)
        if learning:
            self.activation_function_derivative=self.values_vector*(1-self.values_vector)
    def weights_update(self,a):
        self.weights_matrix-=a*(self.delta_l.dot(cp.transpose(self.previous_layer.values_vector)))
    def offsets_update(self,a):
        self.offsets_vector-=a*self.delta_l

class Net():
    icput_layer:InputLayer=None
    hidden_layers:HiddenLayer=[]
    out_layer:OutLayer=None

    def __init__(self,*args):
        self.hidden_layers
        self.icput_layer, *self.hidden_layers, self.out_layer=args
        pl=self.icput_layer
        for layer in [*self.hidden_layers, self.out_layer]:
            layer.connect(pl)
            pl=layer
    
    def forward_propogate(self,icput,learn=False):
        self.icput_layer.set_icput(icput)
        for layer in self.hidden_layers:
            layer.forward_propogate(learn)
        return self.out_layer.output()
    
    def backward_propogate(self,error,i=0):
        layer=self.hidden_layers[i]
        layer.forward_propogate(True)
        if i==len(self.hidden_layers)-1:
            layer.delta_l=error*layer.activation_function_derivative
            return 0
        self.backward_propogate(error,i+1)
        next_layer=self.hidden_layers[i+1]
        layer.delta_l=(cp.transpose(next_layer.weights_matrix)).dot(next_layer.delta_l)*layer.activation_function_derivative

    def train(self,x_train,y_train,alpha=0.01,beta=0,batch=1,iterations=10000,debug=False):
        if debug:
            a=[]
            plt.ion()
            fig=plt.figure()
            ax=fig.add_subplot(111)
            x=cp.zeros((1))
            y=cp.zeros((1))
            line,=ax.plot(x.get(),y.get())
        for iteration in range(iterations):
            print(iteration)
            k=cp.random.randint(len(x_train))
            target=cp.zeros((10,1))
            target[y_train[k]]=1
            error=self.forward_propogate(x_train[k])-target
            self.backward_propogate(error)
            for layer in self.hidden_layers:
                layer.weights_update(alpha)
                layer.offsets_update(beta)
            if debug:
                a.append(cp.linalg.norm(error))
            if debug and iteration%100==0:
                x=cp.linspace(0,iteration+101,iteration+101)
                y=cp.concatenate((cp.array(y),cp.ones((100))*cp.average(a)))
                a=[]
                print(cp.shape(x))
                print(cp.shape(y))
                line.set_xdata(x.get())
                line.set_ydata(y.get())
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

    def print(self):
        for layer in [self.icput_layer,self.hidden_layers,self.out_layer]:
            print(cp.transpose(layer.values_vector))

    def save_config(self,file="config.npz"):
        a=[]
        for layer in self.hidden_layers:
            a.append(layer.weights_matrix)
            a.append(layer.offsets_vector)
        cp.savez_compressed(file,*a)
    
    def load_config(self,file="config.npz"):
        i=0
        for layer in self.hidden_layers:
            layer.weights_matrix=cp.load(file)["arr_"+str(i)]
            layer.offsets_vector=cp.load(file)["arr_"+str(i+1)]
            i+=2