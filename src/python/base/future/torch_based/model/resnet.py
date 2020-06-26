import torch
import torchvision
from src.python.base import TimeDistributed
import sys
import os
import cv2
import torch.nn.functional as F
from src.python.base import resnet50
from torch import nn

import torch.optim as optim


class Resnet_Lipreading(torch.nn.Module):

    def __init__(self):
        super(Resnet_Lipreading,self).__init__()


        self.network = torch.nn.Sequential(
            TimeDistributed(torchvision.models.resnet18(num_classes=2)),
            torch.nn.LSTM(1024, 512),
            torch.nn.LSTM(1024, 512),
            torch.nn.Softmax(2)

        )
    def forward(self,x):
        return self.network(x)
def dataset_loader(path):

    images = []

    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()

    while success:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = image/ 255
        images.append(image)
        success, image = vidcap.read()

    big_window = []
    for i in range(len(images)-4):
        small_window = []
        for j in range(5):
            small_window.append(images[i+j])
        big_window.append(small_window)


    big_window = torch.tensor(big_window)

    big_window = big_window.reshape(big_window.shape[0], big_window.shape[1], big_window.shape[2] , big_window.shape[3])

    return big_window.float()

class Distributed(torch.nn.Module):
    def __init__(self,layer):
        super(Distributed,self).__init__()

        self.layer = layer

    def forward(self,x):
        x = torch.split(x, 1)

        outputs = []
        for i in range(len(x)):
            print(x[i].shape)
            each = torch.reshape(x[i],(x[i].shape[1],x[i].shape[2],x[i].shape[3],x[i].shape[4]))
            each = self.layer(each)
            outputs.append(each)


        return outputs

class Stack(torch.nn.Module):
    def __init__(self):
        super(Stack,self).__init__()

    def forward(self,x):
        x = torch.stack(x,dim=0)
        return x

class Resnet_lipreading(torch.nn.Module):
    def __init__(self,nclasses ):
        super(Resnet_lipreading, self).__init__()

        self.distributed = Distributed(resnet50(num_classes=nclasses))
        self.stack = Stack()

        self.lstm1 = torch.nn.LSTM(2048,512,2,bidirectional=True)
        self.fc1 = torch.nn.Linear(nclasses,2048)
        self.softmax = torch.nn.Softmax(nclasses)

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.distributed(x)
        x = self.stack(x)
        output, (hidden, cell) = self.lstm1(x)

        x = self.fc1(output)
        x = F.softmax(x)


        return x

def resnet():
    module_path = sys.path[1]

    train_set_path = os.path.join(module_path ,'dataset','cut','train')



    train_set = torchvision.datasets.DatasetFolder(root=train_set_path,loader=dataset_loader, extensions='mp4')
    train_loader = torch.utils.data.DataLoader(train_set, 4, shuffle=True)

    nclasses = len(train_set.classes)

    net = Resnet_lipreading(nclasses)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(500):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data



            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return


resnet()







