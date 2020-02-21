#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torchvision
import torch
import torch.nn as nn
import sys
import numpy as np
from torch.autograd import Variable
from sklearn import metrics
data21 = torchvision.datasets.FashionMNIST(' ./', train=False, transform=torchvision.transforms.ToTensor(), target_transform=None, download=True)
print("Department : Computer Science and Automation \nName : Rishabh Ravindra Meshram \nS.R Number : 16149")


# In[ ]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# In[ ]:



loss_fn = torch.nn.CrossEntropyLoss()
def multilayernet():
    num_digits=10
    path = "./model/NN.pth"
    model = torch.load(path)
    model.eval()
    testX=Variable(torch.Tensor.float((testX1)))
    testX=testX.view(-1,784)
    testY1=model(testX)
    loss = loss_fn(testY1, testY)
    print(loss)
    Prediction=list(testY1.max(1)[1].data.tolist())
    file1 = open("multi-layer-net.txt", 'w')
    
    print(len(Prediction))
    print(len(testY))
    count=0
    for i in range(len(testY)):
        if(Prediction[i]==testY[i]):
            count+=1
    acc=((count/len(testY))*100)
    conf_matrix = metrics.confusion_matrix(Prediction, testY)
    print(conf_matrix)
    file1.write("%.2f\n" %acc)
    file1.write("%f\n" %loss)
    file1.write("gt_label,pred_label\n") 
    for i in range(len(testY)):
        file1.write("{},{}\n".format(testY[i], Prediction[i]))
testX1,testY = torch.load("./FashionMNIST/processed/test.pt")    

def Clayernet():
    cnn = CNN()
    num_digits=10
    path = "./model/model_cnn.pth"
    cnn.load_state_dict(torch.load(path))
    cnn.eval()
    test = torch.utils.data.DataLoader(data21, batch_size=len(testY), shuffle=False)
    for (images, labels) in (test):
        images = Variable(images.float())
        outputs = cnn(images)
        Prediction=list(outputs.max(1)[1].data.tolist())
        loss = loss_fn(Prediction, testY)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    #print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))
    file2 = open("convolution-neural-net.txt", 'w')
    print(len(Prediction))
    print(len(testY))
    count=0
    for i in range(len(testY)):
        if(Prediction[i]==testY[i]):
            count+=1
    acc=((count/len(testY))*100)
    conf_matrix = metrics.confusion_matrix(Prediction, testY)
    print(conf_matrix)
    file2.write("%.2f\n" %acc)
    file2.write("%f\n" %loss)
    file2.write("gt_label,pred_label\n") 
    for i in range(len(testY)):
        file2.write("{},{}\n".format(testY[i], Prediction[i]))    

#input2="input1.txt"
multilayernet()
Clayernet()   

