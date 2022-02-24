from matplotlib.cbook import safe_masked_invalid
from NeuroNet import *
import tensorflow as tf
import cupy as np
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
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

x_train=np.array(x_train)
y_train=np.array(y_train)

print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))

net=Net(
    InitLayer(28**2),
    HiddenLayer(500),
    HiddenLayer(500),
    HiddenLayer(10)
)
for i in range(1000000):
    print(i)
    k=np.random.randint(60000)
    x_train_vector=np.array(x_train[k,:,:,0])
    x_train_vector.resize(28**2)
    target=np.zeros(10)
    target[y_train[k]]=1
    net.train(x_train_vector,target,0.01)

n=10
fig,ax=plt.subplots(2,n)
for i in range(n):
    k=np.random.randint(10000)
    x_test_vector=np.array(x_test[i,:,:,0])
    x_test_vector.resize(28**2)

    ax[0][i].imshow(x_test[i,:,:,0])
    ax[0][i].set_yticklabels([])
    ax[0][i].set_xticklabels([])
    temp=net.forward_propogate(x_test_vector)
    temp.resize(10)
    ax[1][i].bar(np.arange(10),temp)
    ax[1][i].set_xticks(np.arange(10),labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    ax[1][i].set_yticklabels([])

result=0
for i in range(10000):
    x_test_vector=np.array(x_test[i,:,:,0])
    x_test_vector.resize(28**2)
    out=net.forward_propogate(x_test_vector)
    if index_of_max(out)==y_test[i]:
        result+=1
result/=10000
plt.show()
print(result)
# input()