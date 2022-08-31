import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from utils import train_model
from aoi_dataset import aoi_Dataset
#import senet

parser = argparse.ArgumentParser(description='PyTorch Implementation of AOI Classifier.')

parser.add_argument('--train_zip', type=str, default='dataset/train_images.zip', help='Training data with zip file path.')
parser.add_argument('--train_csv', type=str, default='dataset/train.csv', help='Training data with csv file path.')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--epoch', type=int, default=30, help='Training epoch')

args = parser.parse_args()

transform = transforms.Compose([transforms.Resize(256), 
                                transforms.RandomCrop(224),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(),
                                transforms.ToTensor()])

                            
train_set = aoi_Dataset(args.train_zip, args.train_csv, transform)
[print('class {0}: {1:.1f}%'.format(i, train_set.get_class_ratio()[i]*100)) for i in range(train_set.num_class)]
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# download pretrained model
model = models.densenet201(pretrained=True)
#model = models.resnet152(pretrained=True)
#model = senet.senet154(num_classes=6)

# modify the input channel and number of class
# ResNet152
#model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=6, bias=True))
# DenseNet201
model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.classifier = nn.Sequential(nn.Linear(in_features=1920, out_features=6, bias=True))

#model = models.DenseNet(growth_rate=32, block_config=(6, 12, 64, 48), num_init_features=64, num_classes=6)
#model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#model.classifier = nn.Sequential(nn.Linear(in_features=1920, out_features=6, bias=True))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
model = train_model(model, criterion, optimizer, train_loader,
                    exp_lr_scheduler, num_epochs=args.epoch, device=device)

train_set.close_zip()
