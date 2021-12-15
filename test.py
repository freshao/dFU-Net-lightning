from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable


# options
dataset = 'cifar10' # options: 'mnist' | 'cifar10'
batch_size = 64   # input batch size for training
epochs = 10       # number of epochs to train
lr = 0.003        # learning rate


# Data Loading
# Warning: this cell might take some time when you run it for the first time,
#          because it will download the datasets from the internet
if dataset == 'mnist':
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST(root='.', train=True, download=True, transform=data_transform)
    testset = datasets.MNIST(root='.', train=False, download=True, transform=data_transform)

elif dataset == 'cifar10':
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = datasets.CIFAR10(root='.', train=True, download=True, transform=data_transform)
    testset = datasets.CIFAR10(root='.', train=False, download=True, transform=data_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


## network and optimizer
if dataset == 'mnist':
    num_inputs = 784
elif dataset == 'cifar10':
    num_inputs = 3072

num_outputs = 10 # same for both CIFAR10 and MNIST, both have 10 classes as outputs

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Net, self).__init__()

        # in_ch = 3
        # out_ch = 16
        # kernel_size = 5
        self.conv1 = nn.Conv2d(in_channels=num_inputs, out_channels=16, kernel_size=5)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=5)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)

        self.linear1 = nn.Linear(3200, 64)
        self.tanh3 = nn.Tanh()
        self.linear2 = nn.Linear(64, num_outputs)

    # b c h w   64 x 1 x 28 x 28 => 64 x 1 x 1 x 784 => 64 x 1 x 1 x 10
    # 64 x 3 x 28 x 28
    # b c d h w
    # 64 x 1 x 100 x 28 x 28

    def forward(self, input, feature=False):
        x1 = self.conv1(input)
        x = x1
        x1 = self.tanh1(x1)
        x1 = self.pool1(x1)

        x2 = self.conv2(x1)
        x2 = self.tanh1(x2)
        x2 = self.pool1(x2)

        batch = input.shape[0]
        x2 = x2.view(batch, -1)
        x3 = self.linear1(x2)
        x3 = self.tanh2(x3)
        x4 = self.linear2(x3)

        output = x4
        if feature:
            return x
        return output

network = Net(num_inputs=3, num_outputs=10)
optimizer = optim.SGD(network.parameters(), lr=lr)


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx == 50 :
            break
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test():
    network.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        #data, target = Variable(data, volatile=True), Variable(target)
        output = network(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        #test_loss += F.cross_entropy(output, target, sum=True).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def imshow(img):
    img = img
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def feature_imshow(img):
    img = img[:3]
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

train(20)

#%%

test()

#%%

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
network.eval()
feature_images = network(images, feature=True).detach()

# show images
imshow(torchvision.utils.make_grid(images))

# show feature
feature_imshow(torchvision.utils.make_grid(feature_images))
