import numpy as np
import sys
import scipy
import torch
import torch.nn.functional as nn
import torchvision as torchvision


def main():
    batch = 100
    epoches = 10
    learning_rate = 0.01
    # option 1 - like ex3.
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    train_x = np.loadtxt(train_x)
    train_y = np.loadtxt(train_y)
    test_x = np.loadtxt(test_x)
    # Convert to tensors
    train_x_tensor = torch.from_numpy(train_x)
    train_y_tensor = torch.from_numpy(train_y)
    test_x_tensor = torch.from_numpy(test_x)

    # option 2 - like Zvika.
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.FashionMNIST(root='./data',
                                                  train=True,
                                                  download=True,
                                                  transform=transforms)
    test_set = torchvision.datasets.FashionMNIST(root='./data',
                                                 train=False,
                                                 download=True,
                                                 transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch,
                                              shuffle=True)

    modelA = FirstNetwork(image_size=28*28)
    optimizer = torch.optim.SGD(modelA.parameters(), lr=learning_rate)
    for i in range(epoches):
        train(i, modelA, train_loader, learning_rate, optimizer)
    test(modelA, test_loader)

    # The difference between the first and second models:
    # On the first network - the optimizer used is SGD.
    # On the second network - the optimizer used is ADAM.
    modelB = FirstNetwork(image_size=28*28)
    optimizer = torch.optim.Adam(modelB.parameters(), lr=learning_rate)
    for i in range(epoches):
        train(i, modelB, train_loader, learning_rate, optimizer)
    test(modelB, test_loader)

    # The third model is just like modelA, but we drop out some of the neurons.


def train(epoch, model, train_loader, learning_rate, optimizer):
    model.train()
    for curr_batch, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    test_loss = 0
    accuracy_counter = 0
    with torch.no_grad:
        for data, target in test_loader:
            output = model(data)  # forward.
            # sum up batch loss and get the index of the max log-probability.
            test_loss += nn.nll_loss(output, target, size_average=False).item()
            prediction = output.max(1, keepdim=True)[1]

            # todo !!!! save predictions to an output file test_y later.
            accuracy_counter += prediction.eq(target.view_as(prediction)).cpu().sum()

    # Get the average loss
    test_loss /= len(test_loader.dataset)
    print('\nTest Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n').format(
        test_loss, accuracy_counter, len(test_loader.dataset), 100. * accuracy_counter / len(test_loader.dataset)
    )


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
        x = x.view(-1, self.image_size)
        x = nn.relu(self.fc0(x))
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        # Dropout! check which 0.4 < p < 0.6 is better.
        m = torch.nn.Dropout(p=0.5)
        input_ = x
        output = m(input_)
        return output
        # return nn.log_softmax(x)


# Two hidden layers.
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
        x = nn.relu(self.fc2(x))
        return nn.log_softmax(x)


# Two hidden layers.
class FirstNetworkWithNormalization(torch.nn.Module):
    def __init__(self, image_size):
        super(FirstNetworkWithNormalization, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):  # Using the activation function - ReLu.
        x = x.view(-1, self.image_size)

        # We must perform Batch Normalization must be before calling relu!
        m = torch.nn.BatchNorm1d(100)  # maybe without learnable parameters???????
        input_ = self.fc0(x)
        output = m(input_)
        # x = nn.relu(self.fc0(x))
        x = nn.relu(output)
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return nn.log_softmax(x)


# todo !!!! notice that on model E, F we don't know which activation function
# todo works best. So, we need to check all possibilities.
# Five hidden layers.
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
        x = nn.relu(self.fc2(x))
        return nn.log_softmax(x)


if __name__ == "__main__":
    main()
