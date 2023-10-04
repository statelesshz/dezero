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


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        return (diff ** 2).sum() / len(diff)
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        return gx0, -gx0


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


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


class Linear(Function):
    """
    Copy from https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/06419d7fb2e7ea19aa3719efc27795edbdc41a1f/dezero/functions.py#L273C1-L289C29
    """
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


class Sigmoid(Function):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)
