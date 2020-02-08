# -------------------------------------------------------------------
# Recurrent Neural Network (RNN)
# -------------------------------------------------------------------

import tensorflow.compat.v1 as tf2
tf2.disable_v2_behavior()
from DToolbox.DBase import DBase
import tensorflow as tf


class DRNN_moon(DBase):
    def __init__(self, input_dim, input_length, output_dim, output_length, cell_size, lr):
        # initial parameters
        self.cell_size = cell_size
        self.x = tf.compat.v1.placeholder(tf.float32, [None, input_dim * input_length])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, output_dim * output_length])

        DBase.__init__(self, input_dim, input_length, output_dim, output_length, lr)


    def RNN(self):
        inputs = tf.reshape(self.x, [-1, self.input_dim, self.input_length])

        # weight and bias
        weights_shape = [self.cell_size, self.output_length]
        self.weights = self.Create_weights(weights_shape)
        biases_shape = [self.output_length]
        self.biases = self.Create_biases(biases_shape)

        # cell type. LSTM, GRU, etc.
        cell_type = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0)
        outputs, final_state = tf.compat.v1.nn.dynamic_rnn(cell_type, inputs, dtype=tf.float32)

        if self.b_Activation == True:
            results = tf.nn.relu(tf.matmul(final_state[1], self.weights) + self.biases)
        else:
            results = tf.matmul(final_state[1], self.weights) + self.biases
        return results


    def Prediction(self):
        # prediction
        self.pred = self.RNN()


    def Get_Loss(self):
        # loss and accuracy
        self.loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.pred)
        self.accuracy = tf.compat.v1.metrics.accuracy(labels=tf.argmax(self.y, axis=1), predictions=tf.argmax(self.pred, axis=1))[1]


    def Optimize(self):
        # optimizer
        self.train_step = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)

