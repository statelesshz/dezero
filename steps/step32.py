if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
from dezero import Variable

def f(x):
    return x ** 4 - 2 * x ** 2

x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph=True)
print(x.grad, type(x.grad))

gx = x.grad
# 二次求导前需要清空 x 原有的导数
x.cleargrad()
gx.backward()
print(x.grad)
