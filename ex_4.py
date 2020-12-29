import numpy as np
import sys
import scipy
import torch
import torch.nn.functional as nn
import torchvision as torchvision
from torch.utils.data import SubsetRandomSampler


def main():
    test_predictions = []
    batch = 64
    valid_ratio = 0.2
    epoches = 10
    learning_rate = 0.001
    """
    # option 1 - like ex3.
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    train_x = np.loadtxt(train_x)
    train_y = np.loadtxt(train_y)
    test_x = np.loadtxt(test_x)
    # Convert to tensors
    train_x_tensor = torch.from_numpy(train_x)
    train_y_tensor = torch.from_numpy(train_y)
    test_x_tensor = torch.from_numpy(test_x)
    """

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.FashionMNIST(root='./data',
                                                  train=True,
                                                  download=True,
                                                  transform=transforms)
    test_set = torchvision.datasets.FashionMNIST(root='./data',
                                                 train=False,
                                                 download=True,
                                                 transform=transforms)

    # Splits train data set to the corresponding validation ratio.
    train_set_size = len(train_set)
    indices = list(range(train_set_size))
    split = int(valid_ratio * train_set_size)
    valid_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(valid_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=batch,
                                                    sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch,
                                              shuffle=False)

    modelA = FirstNetwork(image_size=28*28)
    optimizer = torch.optim.SGD(modelA.parameters(), lr=learning_rate)

    for i in range(epoches):
        print("model A, curr epoch: " + str(i))
        train(i, modelA, train_loader, learning_rate, optimizer)
        validate(modelA, validation_loader)
    test_predictions = test(modelA, test_loader)

    modelB = FirstNetwork(image_size=28*28)
    optimizer = torch.optim.Adam(modelB.parameters(), lr=learning_rate)

    for i in range(epoches):
        print("model B, curr epoch: " + str(i))
        train(i, modelB, train_loader, learning_rate, optimizer)
        validate(modelB, validation_loader)
    test_predictions = test(modelB, test_loader)

    # The third model is just like modelA, but we drop out some of the neurons.
    modelC = FirstNetworkWithDropout(image_size=28*28)
    optimizer = torch.optim.SGD(modelC.parameters(), lr=learning_rate)

    for i in range(epoches):
        print("model C, curr epoch: " + str(i))
        train(i, modelC, train_loader, learning_rate, optimizer)
        validate(modelC, validation_loader)
    test_predictions = test(modelC, test_loader)

    # The fourth model is just like modelA, but we add Batch Normalization
    modelD = FirstNetworkWithNormalization(image_size=28*28)
    optimizer = torch.optim.SGD(modelD.parameters(), lr=learning_rate)

    for i in range(epoches):
        print("model D, curr epoch: " + str(i))
        train(i, modelD, train_loader, learning_rate, optimizer)
        validate(modelD, validation_loader)
    test_predictions = test(modelD, test_loader)
    
    modelE = SecondNetwork(image_size=28*28)
    # Adam is the best optimizer...
    optimizer = torch.optim.Adam(modelE.parameters(), lr=learning_rate)

    for i in range(epoches):
        print("model E, curr epoch: " + str(i))
        train(i, modelE, train_loader, learning_rate, optimizer)
        validate(modelE, validation_loader)
    test_predictions = test(modelE, test_loader)

    modelF = ThirdNetwork(image_size=28*28)
    optimizer = torch.optim.Adam(modelF.parameters(), lr=learning_rate)

    for i in range(epoches):
        print("model F, curr epoch: " + str(i))
        train(i, modelF, train_loader, learning_rate, optimizer)
        validate(modelF, validation_loader)
    test_predictions = test(modelF, test_loader)

    # todo choose the best model and print its output to test_y!


def train(epoch, model, train_loader, learning_rate, optimizer):
    model.train()
    for curr_batch, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def validate(model, validation_loader):
    batch = 64
    model.eval()
    validation_loss = 0
    accuracy_counter = 0
    with torch.no_grad():
        for data, target in validation_loader:
            output = model(data)  # forward.
            # sum up batch loss and get the index of the max log-probability.
            # test_loss += nn.nll_loss(output, target, size_average=False).item()
            validation_loss += nn.nll_loss(output, target, reduction='sum').item()
            prediction = output.max(1, keepdim=True)[1]
            accuracy_counter += prediction.eq(target.view_as(prediction)).cpu().sum()

    # Get the average loss
    validation_loss /= (len(validation_loader) * batch)
    print('\nValidation Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, accuracy_counter, (len(validation_loader) * batch),
        100. * accuracy_counter / (len(validation_loader) * batch)))


def test(model, test_loader):
    prediction_list = []
    model.eval()
    test_loss = 0
    accuracy_counter = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)  # forward.
            # sum up batch loss and get the index of the max log-probability.
            # test_loss += nn.nll_loss(output, target, size_average=False).item()
            test_loss += nn.nll_loss(output, target, reduction='sum').item()
            prediction = output.max(1, keepdim=True)[1]
            prediction_list.append(prediction)
            accuracy_counter += prediction.eq(target.view_as(prediction)).cpu().sum()

    # Get the average loss
    test_loss /= len(test_loader.dataset)
    print('\nTest Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, accuracy_counter, len(test_loader.dataset), 100. * accuracy_counter / len(test_loader.dataset)))

    return prediction_list


def write_to_file(predict_y):
    with open('test_y', 'w') as file:
        for y in predict_y:
            file.write("%s\n" % y)
    file.close()


# Two hidden layers: Model A, B. The difference between the first and second models:
# On the first network - the optimizer used is SGD.
# On the second network - the optimizer used is ADAM.
class FirstNetwork(torch.nn.Module):
    def __init__(self, image_size):
        super(FirstNetwork, self).__init__()
        self.image_size = image_size

        # The 2 hidden layers and the output layer. for each layer:
        #   - the first number = number of input neural.
        #   - the second number = number of output neural.
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):  # Using the activation function - ReLu.
        x = x.view(-1, self.image_size)
        x = nn.relu(self.fc0(x))
        x = nn.relu(self.fc1(x))
        # x = nn.relu(self.fc2(x))
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1)


# Model C - With Dropout
class FirstNetworkWithDropout(torch.nn.Module):
    def __init__(self, image_size):
        super(FirstNetworkWithDropout, self).__init__()
        self.image_size = image_size

        # The 2 hidden layers and the output layer. for each layer:
        #   - the first number = number of input neural.
        #   - the second number = number of output neural.
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):  # Using the activation function - ReLu.
        # Dropout! check which 0.4 < p < 0.6 is better.
        x = x.view(-1, self.image_size)
        x = nn.relu(self.fc0(x))
        m = torch.nn.Dropout(p=0.5)
        x = m(x)

        x = nn.relu(self.fc1(x))
        m = torch.nn.Dropout(p=0.5)
        x = m(x)

        # x = nn.relu(self.fc2(x))
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1)


# Model D - With Batch Normalization
class FirstNetworkWithNormalization(torch.nn.Module):
    def __init__(self, image_size):
        super(FirstNetworkWithNormalization, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc_bn_0 = torch.nn.BatchNorm1d(100)  # Batch Normalization BEFORE the activation function.

        self.fc1 = torch.nn.Linear(100, 50)
        self.fc_bn_1 = torch.nn.BatchNorm1d(50)

        self.fc2 = torch.nn.Linear(50, 10)
        self.fc_bn_2 = torch.nn.BatchNorm1d(10)

    def forward(self, x):  # Using the activation function - ReLu.
        x = x.view(-1, self.image_size)
        x = nn.relu(self.fc_bn_0(self.fc0(x)))
        x = nn.relu(self.fc_bn_1(self.fc1(x)))
        x = nn.relu(self.fc_bn_2(self.fc2(x)))
        return nn.log_softmax(x, dim=1)


# todo !!!! notice that on model E, F we don't know which optimizer works best. So, we need to check all possibilities.
# Five hidden layers: Model E
class SecondNetwork(torch.nn.Module):
    def __init__(self, image_size):
        super(SecondNetwork, self).__init__()
        self.image_size = image_size

        # The 5 hidden layers and the output layer. for each layer:
        #   - the first number = number of input neural.
        #   - the second number = number of output neural.
        self.fc0 = torch.nn.Linear(image_size, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.fc3 = torch.nn.Linear(10, 10)
        self.fc4 = torch.nn.Linear(10, 10)
        self.fc5 = torch.nn.Linear(10, 10)

    def forward(self, x):  # Using the activation function - ReLu.
        x = x.view(-1, self.image_size)
        x = nn.relu(self.fc0(x))
        x = nn.relu(self.fc1(x))
        # x = nn.relu(self.fc2(x))
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1)


# Five hidden layers: Model F
class ThirdNetwork(torch.nn.Module):
    def __init__(self, image_size):
        super(ThirdNetwork, self).__init__()
        self.image_size = image_size

        # The 5 hidden layers and the output layer. for each layer:
        #   - the first number = number of input neural.
        #   - the second number = number of output neural.
        self.fc0 = torch.nn.Linear(image_size, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.fc3 = torch.nn.Linear(10, 10)
        self.fc4 = torch.nn.Linear(10, 10)
        self.fc5 = torch.nn.Linear(10, 10)

    def forward(self, x):  # Using the activation function - Sigmoid.
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1)


if __name__ == "__main__":
    main()
