# -------------------------------------------------------------------
# An example of DCNN by Torch
# -------------------------------------------------------------------

from examples.Mobil_L12.data_io import load_dataset
from DToolbox.TorchCNNs import TCNN
from DToolbox import g_type_DCNN
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import pickle

def cnn_train(model_path=None, b_save=False):
    training_data, training_label, testing_data, testing_label = load_dataset(nn_type=g_type_DCNN, b_shuffle=True)
    print(np.shape(training_data), np.shape(training_label))
    print(np.shape(testing_data), np.shape(testing_label))

    # parameters
    lr = 1e-2

    n_epoch = 10
    batch_size = 50
    n_data = len(training_label)
    n_batch = int(n_data / batch_size)

    net = TCNN()
    # define the loss function
    criterion = nn.CrossEntropyLoss()

    # define the optimizing method
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.1)

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
            xs = xs.view(xs.size()[0], 1, xs.size()[1], xs.size()[2])
            ys = ys.long()
            
            # wrap them in Variable
            inputs, labels = Variable(xs), Variable(ys)            

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            
            outputs = net(inputs)
            # logp = nn.functional.log_softmax(outputs)
            # logpy = torch.gather(logp, 1, labels.view(-1,1))
            # loss = -(logpy).mean()
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
        xs_test = xs_test.view(xs_test.size()[0], 1, xs_test.size()[1], xs_test.size()[2])
        input_test, label_test = Variable(xs_test), Variable(ys_test)
        output_test = net(input_test)
        _, pred_test = torch.max(output_test, 1)
        correct_test = (pred_test == label_test).sum().item()
        total_test = len(label_test)
            
        print("+ {} th, Train: loss={:.4f}. Accuracy={:.4f} %. Test: acc={:.4f} %".
                format(epoch, running_loss,100*correct/total, 100*correct_test/total_test))

    if True == b_save and model_path is not None:
        torch.save(net, model_path)        

if __name__ == '__main__':
    model_path = 'cnn_network/Tcnn.pickle'
    cnn_train(model_path, b_save=True)
