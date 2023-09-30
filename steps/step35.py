if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array(1.0))
y = F.tanh(x)
y.backward(create_graph=True)

for i in range(3):
    gx = x.grad
    print(gx)
    x.cleargrad()
    gx.backward(create_graph=True)
