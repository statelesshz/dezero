import numpy as np

from dezero.core import Function, as_variable
import dezero.utils as utils


class Sin(Function):
    def forward(self, x):
        return np.sin(x)
    
    def backward(self, gy):
        x, = self.inputs
        return gy * cos(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)
    
    def backward(self, gy):
        x, = self.inputs
        return gy * -sin(x)
    

class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, gy):
        y = self.outputs[0]()
        return gy * (1 - y * y)
    

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape)
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)
    

class Transpose(Function):
    def forward(self, x):
        return np.transpose(x)
    
    def backward(self, gy):
        return transpose(gy)
    

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)
    
    def backward(self, gy):
        gy = sum_to(gy, self.x_shape)
        return gy
    

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


class Matmul(Function):
    def forward(self, x, W):
        y = np.dot(x, W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def sin(x):
    return Sin()(x)


def cos(x):
    return Cos()(x)


def tanh(x):
    return Tanh()(x)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


def transpose(x):
    return Transpose()(x)


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x) 


def matmul(x, W):
    return Matmul()(x, W)
