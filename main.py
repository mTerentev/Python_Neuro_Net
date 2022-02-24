import numpy as np
from NeuroNet import *
import tensorflow as tf
import cupy as cp
import matplotlib.pyplot as plt

def index_of_max(vector):
    index=0
    max=0
    i=0
    for a in vector:
        if a>max:
            index=i
            max=a
        i+=1
    return index

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., cp.newaxis]/255.0, x_test[..., cp.newaxis]/255.0

# x_train=x_train[0:1000]
# x_test=x_test[0:100]
# y_train=y_train[0:1000]
# y_test=y_test[0:100]

x_train_prep=cp.reshape(x_train,(len(x_train),28**2))
x_test_prep=cp.reshape(x_test,(len(x_test),28**2))

x_train_prep=cp.array(x_train_prep)
x_test_prep=cp.array(x_test_prep)

y_train=cp.array(y_train)
y_test=cp.array(y_test)

print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))

net=Net(
    InputLayer(28**2),
    HiddenLayer(500),
    HiddenLayer(500),
    HiddenLayer(10),
    OutLayer()
)
#net.train(x_train_prep,y_train,iterations=100000)
#net.save_config()
net.load_config()
# net.train(x_train_prep,y_train,iterations=100000)
# net.save_config()

n=10
fig,ax=plt.subplots(2,n)
for i in range(n):
    k=np.random.randint(len(x_test))

    ax[0][i].imshow(x_test[k])
    ax[0][i].set_yticklabels([])
    ax[0][i].set_xticklabels([])
    temp=net.forward_propogate(x_test_prep[k])
    temp=cp.asnumpy(temp)
    temp.resize(10)
    ax[1][i].bar(np.arange(10),temp)
    ax[1][i].set_xticks(np.arange(10),labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    ax[1][i].set_yticklabels([])


result=0

for i in range(len(x_test)):
    print(i)
    #k=np.random.randint(len(x_test))
    out=net.forward_propogate(x_test_prep[i])
    if index_of_max(out)==y_test[i]:
        result+=1
result/=len(x_test)
plt.show()
print(result)
# icput()