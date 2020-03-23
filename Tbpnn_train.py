# -------------------------------------------------------------------
# An example of DBPNN by Torch
# -------------------------------------------------------------------

from examples.Mobil_L12.data_io import load_dataset
from DToolbox.TorchBPNNs import BPNN
from DToolbox import g_type_DBPNN
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import pickle

def bpnn_train(model_path=None, b_save=False):
    training_data, training_label, testing_data, testing_label = load_dataset(nn_type=g_type_DBPNN, b_shuffle=True)
    print(np.shape(training_data), np.shape(training_label))
    print(np.shape(testing_data), np.shape(testing_label))

    # learning rate
    lr = 0.3

    n_epoch = 100
    batch_size = 50
    n_data = len(training_data)
    n_batch = int(n_data / batch_size)

    net = BPNN()
    # define the loss function
    criterion = nn.CrossEntropyLoss()

    # define the optimizing method
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.5)

    for epoch in range(n_epoch):
        running_loss = 0.0
        correct = 0
        total = 0
        for each_batch in range(n_batch):
            idx = each_batch * batch_size
            if idx + batch_size > n_data:
                break

            # get the training data
            xs = np.asarray(training_data[idx: (idx+batch_size)])
            ys = training_label[idx: (idx+batch_size)]
            ys = [0 if ys[i][0] == 0 else 1 for i in range(len(ys))]            
            ys = np.asarray(ys)
            xs = torch.from_numpy(xs)
            ys = torch.from_numpy(ys)
            # xs = xs.view(xs.size()[0], 1, xs.size()[1], xs.size()[2])
            xs = xs.double()
            ys = ys.long()

            # wrap them in Variable
            inputs, labels = Variable(xs), Variable(ys)            

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # outputs statistics
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.data

        xs_test = np.asarray(testing_data)
        ys_test = testing_label
        ys_test = [0 if ys_test[i][0] == 0 else 1 for i in range(len(ys_test))]
        ys_test = np.asarray(ys_test)
        xs_test = torch.from_numpy(xs_test)
        ys_test = torch.from_numpy(ys_test)
        xs_test = xs_test.double()        
        input_test, label_test = Variable(xs_test), Variable(ys_test)
        output_test = net(input_test)
        _, pred_test = torch.max(output_test, 1)
        correct_test = (pred_test == label_test).sum().item()
        total_test = len(label_test)
        print("+ {} th, Train: loss={:.4f}. Accuracy={:.4f} %. Test: acc={:.4f} %".
                format(epoch, running_loss,100*correct/total, 100*correct_test/total_test))

if __name__ == '__main__':
    model_path = 'bpnn_network/Tbpnn.pickle'
    bpnn_train(model_path, b_save=True)