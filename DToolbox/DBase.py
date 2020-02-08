# -------------------------------------------------------------------
# Base model for all neuron networks
# -------------------------------------------------------------------

import tensorflow as tf


class DBase():
    def __init__(self, input_dim, input_length, output_dim, output_length, lr, b_activation=False):
        # initial parameters
        self.input_dim     = input_dim
        self.input_length  = input_length
        self.output_dim    = output_dim
        self.output_length = output_length
        self.lr            = lr
        self.b_Activation  = b_activation

        self.Prediction()
        self.Get_Loss()
        self.Optimize()

        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    def Create_weights(self, shape):
        return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1))

    def Create_biases(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    # prediction
    def Prediction(self):
        pass

    # loss function
    def Get_Loss(self):
        pass

    # optimizer
    def Optimize(self):
        pass

    # save session and parameters
    def Save(self, sess, path=None):
        if path is None:
            print("!!! Unable to save!")
        self.saver.save(sess, path)

    # restore session and parameters
    def Restore(self, sess, path=None):
        if path is None:
            print("!!! Unable to read!")

        self.saver.restore(sess, path)

