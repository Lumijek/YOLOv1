import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from  matplotlib import patches
import numpy as np
from pprint import pprint
from tqdm import tqdm

from loss import YoloLoss, nms
from model2 import YOLOv1
from dataset import *


# LOSS FUNCTION TAKES IN PREDICTIONS THEN LABELS
#torch.manual_seed(0)
epochs = 150
device = torch.device("mps")
batch_size = 16
classes = ['horse', 'person', 'bottle', 'dog', 'tvmonitor', 'car', 'aeroplane', 'bicycle', 'boat', 'chair', 'diningtable', 'pottedplant', 'train', 'cat', 'sofa', 'bird', 'sheep', 'motorbike', 'bus', 'cow']
dataloader = get_dataset(batch_size)


loss_func = YoloLoss()
model = YOLOv1()
lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.00005, nesterov=True)

def train_network(model, optimizer, criterion, epochs, dataloader, device):
    model = nn.DataParallel(model)
    model = model.to(device)
    cycle = 0
    for epoch in tqdm(range(epochs), desc="Epoch"):
        if epoch == 10:
            optimizer.param_groups[0]["lr"] = 0.005
        if epoch == 80:
            optimizer.param_groups[0]["lr"] = 0.001
        if epoch == 120:
            optimizer.param_groups[0]["lr"] = 0.0001
        for image, label in tqdm(dataloader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            outputs = model(image)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()
            if cycle % 20 == 0:
                print("Loss:", loss.item())
            if cycle % 200 == 0:
                torch.save(model.state_dict(), "model.pt")
            cycle += 1


if __name__ == '__main__':
	train_network(model, optimizer, loss_func, epochs, dataloader, device)		








