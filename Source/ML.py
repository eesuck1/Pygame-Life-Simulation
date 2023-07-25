import numpy


def leaky_relu(x: float, alpha: float = 0.01):
    return numpy.maximum(alpha * x, x)


def softmax(x: numpy.ndarray) -> numpy.ndarray:
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum()
