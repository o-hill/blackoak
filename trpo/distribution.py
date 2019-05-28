'''

    Distribution functions for TRPO agents.

'''

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp



class PDF:
    '''Specifies necessary properties of a distribution.'''

    def __init__(self) -> None:
        pass

    @property
    def parameters(self) -> None:
        '''Return a TF tensor with the appropriate shape for the parameters of the distribution.'''
        raise NotImplementedError

    @parameters.setter
    def parameters(self, value) -> None:
        '''Block setting this property.'''
        raise RuntimeError('Cannot set the parameters property of any distribution!')

    def kl_divergence(self) -> None:
        '''Kullback-Leibler Divergence.'''
        raise NotImplementedError

    def sample(self) -> None:
        '''Sample the distribution.'''
        raise NotImplementedError

    def entropy(self) -> None:
        '''Compute the Shannon Entropy.'''
        raise NotImplementedError


class Categorical(PDF):

    def __init__(self, n_cats: int) -> None:
        '''Parameterize the distribution with the number of categories.'''
        self.n_cats = n_cats

    @property
    def parameters(self) -> tf.Tensor:
        '''Return the TF placeholder for the parameters of the distribution.'''
        return tf.placeholder('float32', [None, self.n_cats])

    def kl_divergence(self, P, Q) -> tf.Tensor:
        '''Returns the Kullback-Leibler Divergence between the two distributions P and Q.'''
        return tf.reduce_sum(P * tf.log(P / Q), axis=1)

    def sample(self, P) -> tf.Tensor:
        '''Sample the distribution with probability P.'''
        n_dim = tf.shape(P)[0]
        return tfp.distributions.categorical.Categorical(P).sample(sample_shape=(n_dim, 1))

    def entropy(self, P) -> tf.Tensor:
        '''Compute the Shannon entropy of the distribution.'''
        return -tf.reduce_sum(P * tf.log(P), axis=1)



if __name__ == '__main__':

    pdf = Categorical(3)
    params = pdf.parameters

    P = np.atleast_2d([0.3, 0.5, 0.2])
    Q = np.atleast_2d([0.4, 0.3, 0.3])

    p_params = pdf.parameters
    q_params = pdf.parameters

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        out = sess.run(pdf.sample(params), feed_dict={ params: P })
        print(out)
        out = sess.run(pdf.kl_divergence(p_params, q_params), feed_dict={ p_params: P, q_params: Q })
        print(out)
        out = sess.run(pdf.entropy(p_params), feed_dict={ p_params: P })
        print(out)

    print('all done')












