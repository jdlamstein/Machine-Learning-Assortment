# Deploy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from cnn import Net
from util import transform
from param_cnn import Param


class Deploy:
    def __init__(self, p):
        self.p = p

    def check_image(self, testloader):
        def imshow(img):
            img = img / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        dataiter = iter(testloader)
        images, labels = next(dataiter)

        # print images
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join(f'{self.p.classes[labels[j]]:5s}' for j in range(4)))

    def check_single_prediction(self, model, images):
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join(f'{self.p.classes[predicted[j]]:5s}'
                                      for j in range(4)))

    def run(self):

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.p.batch_size,
                                                 shuffle=False, num_workers=2)
        modelpath = './cifar_net.pth'
        net = Net()
        net.load_state_dict(torch.load(modelpath))

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in self.p.classes}
        total_pred = {classname: 0 for classname in self.p.classes}

        # again no gradients needed
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.p.classes[label]] += 1
                    total_pred[self.p.classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == '__main__':
    p = Param()
    Dep = Deploy(p)
    Dep.run()
