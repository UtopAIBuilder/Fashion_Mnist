"""
Code to use the saved models for testing
"""

import numpy as np
import pdb
import os
from tqdm import tqdm

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms

from utils import AverageMeter

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5), #(N, 1, 28, 28) -> (N, 6, 24, 24)
            nn.ReLU(),
            nn.AvgPool2d(2, stride = 2), #(N, 6, 24, 24) -> (N, 6, 12, 12)
            
            nn.Conv2d(6, 16, kernel_size = 5), #(N, 6, 12, 12) -> (N, 6, 8, 8)
            nn.ReLU(),
            nn.AvgPool2d(2, stride = 2)) #(N, 6, 8, 8) -> (N, 16, 4, 4)
    
        self.fc_model = nn.Sequential(
            nn.Linear(256, 120), # (N, 256) -> (N, 120)
            nn.ReLU(),
            nn.Linear(120, 84), # (N, 120) -> (N, 84)
            nn.ReLU(),
            nn.Linear(84, 10))  # (N, 84)  -> (N, 10))
            
    def forward(self, x):
        #print(x.shape)
        x = self.cnn_model(x)
        #print(x.shape)
        #print(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc_model(x)
        #print(x.shape)
        return x
    ''' def __init__(self, n_classes=10):
        emb_dim = 20
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.emb = nn.Linear(64*7*7, emb_dim)
        self.clf = nn.Linear(emb_dim, n_classes)

    def num_flat_features(self, x):

        size = x.size()[1:]  # All dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.emb(x)
        out = self.clf(x)

        return out'''

class MLP(nn.Module):

    def __init__(self, n_classes=10):
        '''
        Define the initialization function of LeNet, this function defines
        the basic structure of the neural network
        '''

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)
        #self.fc3 = nn.Linear(500, 16)
        #self.drop=nn.Dropout(.6)
        self.clf = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        #x=self.drop(x)
        x = F.relu(self.fc2(x))
        #x=self.drop(x)
        #x = F.relu(self.fc3(x))
        out = self.clf(x)

        return out


def test(model, testloader):
    """ Training the model using the given dataloader for 1 epoch.

    Input: Model, Dataset, optimizer,
    """

    model.eval()
    avg_loss = AverageMeter("average-loss")

    y_gt = []
    y_pred_label = []

    for batch_idx, (img, y_true) in enumerate(testloader):
        img = Variable(img)
        y_true = Variable(y_true)
        out = model(img)
        y_pred = F.softmax(out, dim=1)
        y_pred_label_tmp = torch.argmax(y_pred, dim=1)

        loss = F.cross_entropy(out, y_true)
        avg_loss.update(loss, img.shape[0])

        # Add the labels
        y_gt += list(y_true.numpy())
        y_pred_label += list(y_pred_label_tmp.numpy())

    return avg_loss.avg, y_gt, y_pred_label


if __name__ == "__main__":

    trans_img = transforms.Compose([transforms.ToTensor()])
    dataset = FashionMNIST("./data/", train=False, transform=trans_img, download=True)
    testloader = DataLoader(dataset, batch_size=1024, shuffle=False)

    #from train_multi_layer import MLP
    model_MLP = MLP(10)
    model_MLP.load_state_dict(torch.load("./models/MLP.pt"))

    #from training_conv_net import LeNet
    model_conv_net = LeNet()
    model_conv_net.load_state_dict(torch.load("./models/convNet.pt"))

    loss, gt, pred = test(model_MLP, testloader)
    with open("multi-layer-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))

    loss, gt, pred = test(model_conv_net, testloader)
    with open("convolution-neural-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))
