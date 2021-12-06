import numpy as np
from numpy import random
from numpy.core.fromnumeric import transpose
a=random.random(3)
b=random.random(3)
sgm=lambda x: 1/(1+np.exp(-x))
print(np.asfarray(sgm(np.cross(a,b))))
# print([np.asarray_chkfinite(a)np.asarray_chkfinite(b)])
