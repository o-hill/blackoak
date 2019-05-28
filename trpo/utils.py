'''

    Utilities for TRPO computations.

'''

import numpy as np
import tensorflow as tf
from functools import reduce


def numel(tensor: tf.Tensor) -> int:
    '''Calculate the total number of parameters in a tensor.'''
    return reduce(lambda x, y: x * y, tensor_shape(tensor))


def tensor_shape(tensor: tf.Tensor) -> tuple:
    '''Return a well-defined tensor shape. Throws errors on unknown dimensions.'''

    shape = tensor.get_shape()

    # Block unknown dimensions.
    assert all(isinstance(s, int) for s in shape)
    return shape


class FlattenTheta:

    def __init__(self, session: tf.Session, params: tf.Tensor) -> None:
        '''Flatten a TF tensor into a single dimensional vector.'''
        self.session = session
        self.tf_op = tf.concat([tf.reshape(layer, numel(layer)) for layer in params])

    def __call__(self) -> tf.Tensor:
        '''Evaluate the flattening operation.'''
        return self.tf_op.eval(session=self.session)


class IntegrateTheta:

    def __init__(self, session: tf.Session, params: tf.Tensor) -> None:
        '''Replace the weights in a network with the updated weights.'''
        self.session = session

        shapes = [*map(var_shape, params)]
        length1d = np.sum([np.prod(shape) for shape in shapes])

        self.theta = tf.placeholder('float32', [length1d])

        # Reshape parameters into W matrices and b vectors appropriately.
        ops, start = [], 0
        for shape, params in zip(shapes, params):
            length = reduce(lambda x, y: x * y, shape)
            ops.append(param.assign(tf.reshape(self.theta[start : start + length], shape)))
            start += length

        self.op = tf.group(*ops)


    def __call__(self, theta: tf.Tensor) -> tf.Tensor:
        '''Reintegrate the updated weights in the flat theta tensor.'''
        self.session.run(self.op, feed_dict={ self.theta: theta })

















