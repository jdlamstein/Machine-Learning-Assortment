import torch
import torch.optim as optim
import torchvision
from cnn import Net
from param_cnn import Param
from util import transform
import torch.nn as nn


class Train:
    def __init__(self, p):
        self.p = p

    def run(self):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.p.batch_size,
                                                  shuffle=True, num_workers=2)

        net = Net()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.p.learning_rate, momentum=self.p.momentum)

        for epoch in range(self.p.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
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
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')

        modelpath = './cifar_net.pth'
        torch.save(net.state_dict(), modelpath)


if __name__ == '__main__':
    p = Param()
    Tr = Train(p)
    Tr.run()
