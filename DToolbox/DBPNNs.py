# -------------------------------------------------------------------
# Back Propagation Neuron Network (BPNN)
# -------------------------------------------------------------------

import tensorflow.compat.v1 as tf2
tf2.disable_v2_behavior()
from DToolbox.DBase import DBase
import tensorflow as tf


class DBPNN_moon(DBase):
    def __init__(self, input_dim, input_length, output_dim, output_length, lr):
        # input
        self.x = tf.compat.v1.placeholder(tf.float32, [None, input_dim * input_length])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, output_dim * output_length])
        DBase.__init__(self, input_dim, input_length, output_dim, output_length, lr)

    # fc layere
    def NN(self):
        inputs = tf.reshape(self.x, [-1, self.input_dim * self.input_length])
        # initial weight and bias
        weights_shape = [self.input_dim * self.input_length,
                         self.output_dim * self.output_length]
        biases_shape = [self.output_dim * self.output_length]
        self.weights = self.Create_weights(weights_shape)
        self.biases  = self.Create_biases(biases_shape)

        if self.b_Activation ==  True:
            return tf.nn.relu(tf.add(tf.matmul(inputs, self.weights), self.biases))
        else:
            return tf.add(tf.matmul(inputs, self.weights), self.biases)

    # prediction
    def Prediction(self):
        self.pred = self.NN()

    def Get_Loss(self):
        # loss and accuracy
        self.loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.pred)
        self.accuracy = tf.compat.v1.metrics.accuracy(labels=tf.argmax(self.y, axis=1), predictions=tf.argmax(self.pred, axis=1))[1]

    def Optimize(self):
        # optimizer
        self.train_step = tf.compat.v1.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        # self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
