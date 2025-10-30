import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets import Datsets, transforms
from torch.utils.data import DataLoader

#Transforms
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
#Load the dataset
train_data=datasets.CIFAR100(root='./dir',train=True,download=True,transform=transform)
test_data=datasets.CIFAR100(root='./dir',train=False,download=True,transform
=transform)
#Data loaders
train_loader=DataLoader(train_data,batch_size=128,shuffle=True)
test_loader=DataLoader(test_data,batch_size=128,shuffle=False)
#Architechture
class CNN(nn.Module):