# -------------------------------------------------------------------
# An example of DRNN
# -------------------------------------------------------------------


from examples.Mobil_L12.data_io import load_dataset
from DToolbox.DRNNs import DRNN_moon
from DToolbox import g_type_DRNN
import tensorflow as tf
import numpy as np


def drnn_train_moon(model_path=None, b_save=False):

    training_data, training_label, testing_data, testing_label = load_dataset(nn_type=g_type_DRNN, b_shuffle=True)
    print(np.shape(training_data), np.shape(training_label))
    print(np.shape(testing_data), np.shape(testing_label))

    # parameters
    input_dim     = 1
    output_dim    = 1
    input_length  = len(training_data[0])
    output_length = len(training_label[0])
    lr            = 1e-2
    cell_size     = 200

    moon = DRNN_moon(input_dim, input_length, output_dim, output_length, cell_size, lr)

    n_epoch = 20
    batch_size = 50
    n_data = len(training_label)
    n_batch = int(n_data / batch_size)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        test_xs = testing_data
        test_ys = testing_label
        test_loss, test_acc = sess.run([moon.loss, moon.accuracy],
                                       feed_dict={moon.x: test_xs, moon.y: test_ys})
        print("+ Initial! Test: loss={:.5f}, acc={:.2f} %".format(test_loss, test_acc * 100))

        for each_epoch in range(n_epoch):
            for each_batch in range(n_batch):
                idx = each_batch * batch_size

                if idx + batch_size > n_data:
                    break

                xs = training_data[idx: (idx + batch_size)]
                ys = training_label[idx: (idx + batch_size)]

                _, train_loss, train_pred, train_acc = sess.run([moon.train_step, moon.loss,
                                                                 moon.pred, moon.accuracy],
                                               feed_dict={moon.x: xs, moon.y: ys})

            test_loss, test_acc = sess.run([moon.loss, moon.accuracy],
                                           feed_dict={moon.x: test_xs, moon.y: test_ys})

            print("+ {} th, Train: loss={}, acc={:.2f} %. Test: loss={}, acc={:.2f} %".
                  format(each_epoch, train_loss, train_acc*100, test_loss, test_acc*100))


        if True == b_save and model_path is not None:
            moon.Save(sess, model_path)

        return


if __name__ == '__main__':
    drnn_train_moon()


