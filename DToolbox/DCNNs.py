# -------------------------------------------------------------------
# Convolutional Neural Network (CNN)
# -------------------------------------------------------------------


import tensorflow.compat.v1 as tf2
tf2.disable_v2_behavior()
from DToolbox.DBase import DBase
import tensorflow as tf
import numpy as np

class DCNN_moon(DBase):
    '''
    DCNN_moon, two convolutional layers, kernels=[1,2,2,1], single fc network.
    '''
    def __init__(self, input_dim, input_length, output_dim, output_length, lr, input_component, b_activation=False):

        # initial parameters
        self.x = tf.compat.v1.placeholder(tf.float32, [None, input_dim, input_length * input_component])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, output_dim * output_length])
        self.input_component = input_component  # single component 1, three component 3
        self.output_length = output_length
        DBase.__init__(self, input_dim, input_length, output_dim, output_length, lr, b_activation)


    def Create_convolutional_layer(self, input, num_input_channels,
                                   kernal_size, num_kernal,
                                   b_pooling=True):

        weights_shape = [kernal_size, kernal_size, num_input_channels, num_kernal]
        biases_shape = [num_kernal]
        weights = self.Create_weights(shape=weights_shape)
        biases = self.Create_biases(shape=biases_shape)
        layer = tf.nn.conv2d(input=input,
                             filters=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        layer = tf.add(layer, biases)
        if b_pooling == True:
            layer = tf.compat.v1.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

        if self.b_Activation is True:
            layer = tf.nn.relu(layer)
            # layer = tf.nn.softmax(layer)

        return layer, weights


    def Flatten(self, layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])

        return layer_flat, num_features


    def Create_fc_layer(self, input,
                        num_inputs,
                        num_outputs,
                        b_activation):

        weights_shape = [num_inputs, num_outputs]
        weights = self.Create_weights(shape=weights_shape)
        biases_shape = [num_outputs]
        biases = self.Create_biases(shape=biases_shape)
        layer = tf.matmul(input, weights) + biases

        if b_activation == True:
            # layer = tf.nn.relu(layer)
            layer = tf.nn.softmax(layer)

        return layer, weights

    def Prediction(self):
        self.pred = self.CNN()
        # self.pred_fabs = tf.math.abs(self.pred)
        # self.pred_prob = tf.sigmoid(self.pred)


    def Get_Loss(self):
        print("y=", self.y)
        print("pred=", self.pred)
        self.loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.pred)

        # accuracy
        self.accuracy = tf.compat.v1.metrics.accuracy(labels=tf.argmax(self.y, axis=1),
                                            predictions=tf.argmax(self.pred, axis=1))[1]

    def Optimize(self):
        self.train_step =tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)


    def CNN(self):
        # 2 CL + 2 FC layers.
        input = tf.reshape(self.x, [-1, self.input_dim, self.input_length, self.input_component])
        kernal_size_1 = 2
        num_kernal_1 = 1
        layer_conv1, weights_conv1 = self.Create_convolutional_layer(input=input,
                                                                     num_input_channels=self.input_component,
                                                                     kernal_size=kernal_size_1,
                                                                     num_kernal=num_kernal_1,
                                                                     b_pooling=True)
        print("after conv1=", np.shape(layer_conv1))

        kernal_size_2 = 2
        num_kernal_2 = 1
        layer_conv2, weights_conv2 = self.Create_convolutional_layer(input=layer_conv1,
                                                                     num_input_channels=num_kernal_1,
                                                                     kernal_size=kernal_size_2,
                                                                     num_kernal=num_kernal_2,
                                                                     b_pooling=True)
        print("after conv2=", np.shape(layer_conv2))

        # flatting
        layer_flat, num_features = self.Flatten(layer_conv2)
        print("num_features=", num_features)

        # use one fc layer for publication
        output_size = self.output_length
        self.fc_output, self.weight = self.Create_fc_layer(input=layer_flat,
                                                            num_inputs=num_features,
                                                            num_outputs=output_size,
                                                            b_activation=False)
    
        if self.b_Activation == True:
            return tf.nn.softmax(self.fc_output)
        else:
            return self.fc_output


class DCNN_earth(DCNN_moon):
    def CNN(self):
        # 2 Convolutional layers with maxmum pooling layer.
        # The first has 5 convolutional kernels, the second has 5 kernels.
        # convolution kernel shape = [1, 2, 2, 1], pooling kernel = [1, 2, 2, 1]
        # # #
        # 2 Fully connected layers
        # The first matrix is [n_feature, 128], the second matrix is [128, 2].

        input = tf.reshape(self.x, [-1, self.input_dim, self.input_length, self.input_component])
        kernal_size_1 = 2
        num_kernal_1 = 5
        layer_conv1, weights_conv1 = self.Create_convolutional_layer(input=input,
                                                                     num_input_channels=self.input_component,
                                                                     kernal_size=kernal_size_1,
                                                                     num_kernal=num_kernal_1,
                                                                     b_pooling=True)
        print("after conv1=", np.shape(layer_conv1))

        kernal_size_2 = 2
        num_kernal_2 = 5
        layer_conv2, weights_conv2 = self.Create_convolutional_layer(input=layer_conv1,
                                                                     num_input_channels=num_kernal_1,
                                                                     kernal_size=kernal_size_2,
                                                                     num_kernal=num_kernal_2,
                                                                     b_pooling=True)
        print("after conv2=", np.shape(layer_conv2))

        # flatting
        layer_flat, num_features = self.Flatten(layer_conv2)
        print("num_features=", num_features)


        # fc layer 1
        cell_size = 128
        fc_size = cell_size
        fc_layer_1, self.weight1 = self.Create_fc_layer(input=layer_flat,
                                          num_inputs=num_features,
                                          num_outputs=fc_size,
                                          b_activation=False)
        # fc layer 2
        self.fc_output, self.weight2 = self.Create_fc_layer(input=fc_layer_1,
                                              num_inputs=fc_size,
                                              num_outputs=self.output_length,
                                              b_activation=False)

        if self.b_Activation == True:
            # tf.nn.relu(self.fc_output)
            return tf.nn.softmax(self.fc_output)
        else:
            return self.fc_output



class DCNN_Spaceship(DCNN_moon):
    def CNN(self):
        # 2 Convolutional layers with maxmum pooling layer.
        # The first has 10 convolutional kernels, the second has 10 kernels.
        # convolution kernel shape = [1, 2, 2, 1], pooling kernel = [1, 2, 2, 1]
        # # #
        # 2 Fully connected layers with using activation function.
        # The first matrix is [n_feature, 360], the second matrix is [360, 2].

        input = tf.reshape(self.x, [-1, self.input_dim, self.input_length, self.input_component])
        kernal_size_1 = 2
        num_kernal_1 = 10
        # self.b_Activation =True
        layer_conv1, weights_conv1 = self.Create_convolutional_layer(input=input,
                                                                     num_input_channels=self.input_component,
                                                                     kernal_size=kernal_size_1,
                                                                     num_kernal=num_kernal_1,
                                                                     b_pooling=True)
        print("after conv1=", np.shape(layer_conv1))

        kernal_size_2 = 2
        num_kernal_2 = 10
        layer_conv2, weights_conv2 = self.Create_convolutional_layer(input=layer_conv1,
                                                                     num_input_channels=num_kernal_1,
                                                                     kernal_size=kernal_size_2,
                                                                     num_kernal=num_kernal_2,
                                                                     b_pooling=True)
        print("after conv2=", np.shape(layer_conv2))

        # flatting
        layer_flat, num_features = self.Flatten(layer_conv2)
        print("num_features=", num_features)


        # fc layer 1
        cell_size = 360
        fc_size = cell_size
        fc_layer_1, self.weight1 = self.Create_fc_layer(input=layer_flat,
                                          num_inputs=num_features,
                                          num_outputs=fc_size,
                                          b_activation=False)
        # fc layer 2
        self.fc_output, self.weight2 = self.Create_fc_layer(input=fc_layer_1,
                                              num_inputs=fc_size,
                                              num_outputs=self.output_length,
                                              b_activation=False)

        if self.b_Activation == True:
            # tf.nn.relu(self.fc_output)
            return tf.nn.softmax(self.fc_output)
        else:
            return self.fc_output


class DCNN_fivestar(DCNN_moon):
    def CNN(self):
        # 2 Convolutional layers with maxmum pooling layer.
        # The first has 10 convolutional kernels, the second has 10 kernels.
        # convolution kernel shape = [1, 2, 2, 1], pooling kernel = [1, 2, 2, 1]
        # # #
        # 2 Fully connected layers with using activation function.
        # The first matrix is [n_feature, 360], the second matrix is [360, 2].

        input = tf.reshape(self.x, [-1, self.input_dim, self.input_length, self.input_component])
        # self.b_Activation =True

        # the first layer.
        kernal_size_1 = 2
        num_kernal_1 = 10
        layer_conv1, weights_conv1 = self.Create_convolutional_layer(input=input,
                                                                     num_input_channels=self.input_component,
                                                                     kernal_size=kernal_size_1,
                                                                     num_kernal=num_kernal_1,
                                                                     b_pooling=True)
        print("after conv1=", np.shape(layer_conv1))

        # the second layer.
        kernal_size_2 = 2
        num_kernal_2 = 10
        layer_conv2, weights_conv2 = self.Create_convolutional_layer(input=layer_conv1,
                                                                     num_input_channels=num_kernal_1,
                                                                     kernal_size=kernal_size_2,
                                                                     num_kernal=num_kernal_2,
                                                                     b_pooling=True)
        print("after conv2=", np.shape(layer_conv2))

        # the third layer.
        kernal_size_3 = 2
        num_kernal_3 = 10
        layer_conv3, weights_conv3 = self.Create_convolutional_layer(input=layer_conv2,
                                                                     num_input_channels=num_kernal_2,
                                                                     kernal_size=kernal_size_3,
                                                                     num_kernal=num_kernal_3,
                                                                     b_pooling=True)
        print("after conv3=", np.shape(layer_conv3))

        # the fourth layer.
        kernal_size_4 = 2
        num_kernal_4 = 10
        layer_conv4, weights_conv4 = self.Create_convolutional_layer(input=layer_conv3,
                                                                     num_input_channels=num_kernal_3,
                                                                     kernal_size=kernal_size_4,
                                                                     num_kernal=num_kernal_4,
                                                                     b_pooling=True)
        print("after conv3=", np.shape(layer_conv4))

        # the fifth layer.
        kernal_size_5 = 2
        num_kernal_5 = 10
        layer_conv5, weights_conv5 = self.Create_convolutional_layer(input=layer_conv4,
                                                                     num_input_channels=num_kernal_4,
                                                                     kernal_size=kernal_size_5,
                                                                     num_kernal=num_kernal_5,
                                                                     b_pooling=True)
        print("after conv3=", np.shape(layer_conv5))

        # flatting
        layer_flat, num_features = self.Flatten(layer_conv4)
        print("num_features=", num_features)

        # fc layer 1
        cell_size = 360
        fc_size = cell_size
        fc_layer_1, self.weight1 = self.Create_fc_layer(input=layer_flat,
                                                        num_inputs=num_features,
                                                        num_outputs=fc_size,
                                                        b_activation=False)
        # fc layer 2
        self.fc_output, self.weight2 = self.Create_fc_layer(input=fc_layer_1,
                                                            num_inputs=fc_size,
                                                            num_outputs=self.output_length,
                                                            b_activation=False)

        if self.b_Activation == True:
            # tf.nn.relu(self.fc_output)
            return tf.nn.softmax(self.fc_output)
        else:
            return self.fc_output

if __name__ == '__main__':
    pass

