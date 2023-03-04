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
epochs = 10
device = torch.device("mps")
batch_size = 1
classes = ['horse', 'person', 'bottle', 'dog', 'tvmonitor', 'car', 'aeroplane', 'bicycle', 'boat', 'chair', 'diningtable', 'pottedplant', 'train', 'cat', 'sofa', 'bird', 'sheep', 'motorbike', 'bus', 'cow']
dataloader = get_dataset(batch_size)


loss_func = YoloLoss()
model = YOLOv1()
model = nn.DataParallel(model)
cp = torch.load("data/model.pt", map_location=torch.device('cpu'))
model.load_state_dict(cp)
model.eval()
lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.00005, nesterov=True)


i, o = next(iter(dataloader))
#i2, o2 = next(iter(dataloader))
#i = torch.rand_like(i)
l1 = model(i)
a = torch.rand(7, 7, 30)
boxes = nms(l1[0], 0.5, 0.6)
print(boxes.shape)
#l2 = model(i2)
#show_image((i, l1))
#l1 = l1[0]
#l2 = l2[0]
#l1 = l1[l1[:, :, :, 4] > 0.99]
#print(l1.shape)
#l2 = l2[l2[:, :, 4] > 0.99]
#print(l1[0])
#print(l1[1])
show_image((i, l1))
#print(l2)

'''
def train_network(model, optimizer, criterion, epochs, dataloader, device):
	model = model.to(device)
	cycle = 0
	for epoch in tqdm(range(epochs), desc="Epoch"):
		for image, label in tqdm(dataloader):
			image = image.to(device)
			label = label.to(device)

			optimizer.zero_grad()

			outputs = model(image)
			loss = criterion(outputs, label)

			loss.backward()
			optimizer.step()
			if cycle % 10 == 0:
				print("Loss:", loss.item())
			cycle += 1


if __name__ == '__main__':
	pass

	#train_network(model, optimizer, loss_func, epochs, dataloader, device)		
'''








