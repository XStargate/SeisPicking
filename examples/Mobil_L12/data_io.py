# -------------------------------------------------------------------
# Data I/O
# -------------------------------------------------------------------


import pickle
import os
import numpy as np
from DToolbox import g_type_DRNN, g_type_DCNN,g_type_DBPNN, g_DMINIMUM_VALUE


def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data

def generate_label(length, num_cluster, cluster_id):
    if cluster_id >= num_cluster:
        print("Cluster_id must be less than num_cluster")
        return None

    label = np.zeros([int(length), int(num_cluster)])
    label[:, int(cluster_id)] = 1
    return label


def split_data(data, label, split_rate=0.8):
    length = len(data)
    if length != len(label):
        print("Label does NOT match data!")
        return

    index    = int(length * split_rate)
    dat_1a   = data[:index, :]
    label_1a = label[:index, :]
    dat_2b   = data[index:, :]
    label_2b = label[index:, :]

    return dat_1a, label_1a, dat_2b, label_2b


def shuffle_data(data, label):
    length = len(data)
    rand_idx = np.arange(0, length, 1)
    np.random.shuffle(rand_idx)

    return data[rand_idx, :], label[rand_idx, :]



def read_Mobil_dataset(b_shuffle=True):
    data_dir = os.getcwd()+'/data/'
    neg_dir  = "negative/"
    pos_dir  = "positive/"

    negative_file = str(data_dir)+str(neg_dir) + str("negative") + str(".pickle")
    positive_file = str(data_dir) + str(pos_dir) + str("positive") + str(".pickle")

    # read data
    neg_data = read_pickle(negative_file)
    pos_data = read_pickle(positive_file)


    # create label - onehot, [1, 0]--noise, [0, 1]--first break
    num_cluster = 2
    neg_cluster_id = 0
    pos_cluster_id = 1
    neg_label = generate_label(len(neg_data), num_cluster, neg_cluster_id)
    pos_label = generate_label(len(pos_data), num_cluster, pos_cluster_id)

    # split data
    split_rate = 0.8
    neg_train_data, neg_train_label, neg_test_data, neg_test_label = split_data(neg_data, neg_label, split_rate)
    pos_train_data, pos_train_label, pos_test_data, pos_test_label = split_data(pos_data, pos_label, split_rate)

    # merge data
    train_data  = np.vstack([neg_train_data, pos_train_data])
    train_label = np.vstack([neg_train_label, pos_train_label])
    test_data   = np.vstack([neg_test_data, pos_test_data])
    test_label  = np.vstack([neg_test_label, pos_test_label])

    # shuffle
    if b_shuffle is True:
        train_data, train_label = shuffle_data(train_data, train_label)
        test_data, test_label   = shuffle_data(test_data, test_label)

    return train_data, train_label, test_data, test_label


def wave2image(wave, counts, Threshold=None):
    length = len(wave)
    counts = int(counts)
    num_counts = int(counts)
    image = np.zeros([2 * num_counts, length])
    if Threshold is None:
        resolution = np.max(np.fabs(wave)) / num_counts
    else:
        resolution = np.fabs(Threshold) / num_counts

    offset = 0
    for idx in range(length):
        sample = wave[idx]
        if np.fabs(sample) < g_DMINIMUM_VALUE:
            continue

        amplitude = int((sample-offset)/resolution)
        if sample < 0:
            image[amplitude+counts:counts, idx] = 1
        else:
            image[counts:counts+amplitude, idx] = 1

    return image


def get_DCNN_database(b_shuffle=True):
    train_data, train_label, test_data, test_label = read_Mobil_dataset(b_shuffle=b_shuffle)
    print("+ Loading waveform as image... ")

    train_images = []
    test_images = []
    length = len(train_data)
    counts = 100
    for idx in range(length):
        wave = train_data[idx]
        train_images.append(wave2image(wave, counts))
    # train_images = np.vstack(train_images)

    length = len(test_data)
    for idx in range(length):
        wave = test_data[idx]
        test_images.append(wave2image(wave, counts))
    # test_images = np.vstack(test_images)

    return train_images, train_label, test_images, test_label


def load_dataset(nn_type, b_shuffle=False):
    if nn_type == g_type_DRNN or nn_type == g_type_DBPNN:
        return read_Mobil_dataset(b_shuffle=b_shuffle)
    elif nn_type is g_type_DCNN:
        return get_DCNN_database(b_shuffle)



if __name__ == '__main__':
    pass
    # train_images, train_label, test_images, test_label = get_DCNN_database()
    # print(np.shape(train_images))

    # import matplotlib.pyplot as plt
    # plt.imshow(train_images[0])
    # plt.show()









