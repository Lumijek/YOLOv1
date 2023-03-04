import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
from matplotlib import patches
from pprint import pprint
from time import perf_counter as pf

class YoloDataset(Dataset):
	def __init__(self, data, S=7, B=2, C=20, image_size=448):
		super().__init__()
		self.S = S
		self.B = B
		self.C = C
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		# output (B, S, S, B * 5 + C)
		# where last dimension is like
		# [x, y, w, h, I(O), 0, 0, 0, 0, 0 Pr(C0), Pr(C1) ... Pr(19)] # 0s so that output shape and label shape matchup to 30
		# I(O) is binary indicator function if object is present

		i, o = self.data[idx]

		label = torch.zeros(self.S, self.S, self.B * 5 + self.C)
		bnds = o['annotation']['object']
		xscale = image_size / int(o['annotation']['size']['width'])
		yscale = image_size / int(o['annotation']['size']['height'])

		for bnd in bnds:
			name = classes.index(bnd["name"])
			one_hot = F.one_hot(torch.tensor(name), self.C)
			b = bnd['bndbox']
			xmin = (int(b['xmin']) * xscale) / image_size
			ymin = (int(b['ymin']) * yscale) / image_size
			width = ((int(b['xmax']) - int(b['xmin'])) * xscale) / image_size # width and height are relative to whole image 
			height = ((int(b['ymax']) - int(b['ymin'])) * yscale) / image_size # so we done have to scale them

			xcenter = xmin + width / 2
			ycenter = ymin + height / 2

			# now need to get [x, y, w, h] in terms of specific cell center is in (i, j)

			i, j = int(xcenter * self.S), int(ycenter * self.S)
			tx, ty = xcenter * self.S - i, ycenter * self.S - j # we do need to rescale xcenter and ycenter however

			# also only add this class if its only one in the cell cuz yolov1 sucks
			if label[i, j, 4] == 0: # check if cell occupies object
				label[i, j] = torch.cat([torch.tensor([tx, ty, width, height, 1, 0, 0, 0, 0, 0]), one_hot]) # must keep shape

		return self.data[idx][0], label

classes = ['horse', 'person', 'bottle', 'dog', 'tvmonitor', 'car', 'aeroplane', 'bicycle', 'boat', 'chair', 'diningtable', 'pottedplant', 'train', 'cat', 'sofa', 'bird', 'sheep', 'motorbike', 'bus', 'cow']
image_size = 448

@torch.no_grad()
def show_image(inp):
	fig, ax = plt.subplots()
	i, o = inp
	o = o[0]
	S = o.shape[0]
	i = i[0].permute(1, 2, 0).numpy()
	outs = o[o[:, :, 4] > 0.993]
	print(outs.shape)
	for o2 in outs:

		o2[4:] += 1e-4
		a = (o == o2).nonzero(as_tuple=True)[:2]
		si, sj = torch.mode(a[0]).values, torch.mode(a[1]).values

		xmin = float(((o2[0] + si) / S) - o2[2] / 2)
		ymin = float(((o2[1] + sj) / S) - o2[3] / 2)
		width = float(o2[2])
		height = float(o2[3])
		rect = patches.Rectangle((xmin * image_size , ymin * image_size), width * image_size, height * image_size, linewidth=1, edgecolor='r', facecolor='none')
		ax.add_patch(rect)

	ax.imshow(i)
	plt.show()

transform = transforms.Compose(
	[
	transforms.Resize((image_size, image_size)),
	transforms.ColorJitter(0.4, 0.4, 0.4, hue=0.02),
	transforms.ToTensor(),
	]
)

def get_dataset(batch_size=16, S=7, B=2, C=20, image_size=448):
	data = torchvision.datasets.VOCDetection("data", '2012', 'train', download=False, transform=transform)
	dataloader = DataLoader(YoloDataset(data, S, B, C, image_size), batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
	return dataloader
