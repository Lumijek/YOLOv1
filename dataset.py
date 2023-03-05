import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
from matplotlib import patches
from pprint import pprint
from time import perf_counter as pf
from loss import iou

classes = ['horse', 'person', 'bottle', 'dog', 'tvmonitor', 'car', 'aeroplane', 'bicycle', 'boat', 'chair', 'diningtable', 'pottedplant', 'train', 'cat', 'sofa', 'bird', 'sheep', 'motorbike', 'bus', 'cow']
image_size = 448


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

@torch.no_grad()
def nms(boxes, conf_threshold, iou_threshold):
	boxes = boxes.detach().clone()
	# boxes of shape (S, S, 30)
	S = boxes.shape[0]

	cr = torch.arange(S * S) // S
	cc = torch.arange(S * S) % S 
	boxes = boxes.view(-1, 30)
	boxes[:, 0] = (boxes[:, 0] + cr) / S - boxes[:, 2] / 2 # convert x and y from respect to cell
	boxes[:, 1] = (boxes[:, 1] + cc) / S - boxes[:, 3] / 2 # to respect to whole image
	bclass = torch.max(boxes[:, 10:], dim=1)[1] # get class of object
	boxes = torch.cat([boxes[:, 0:5], boxes[:, 5:10]], dim=0) # concatenate both bounding box predictions
	boxes = torch.cat([boxes, bclass.repeat(2).unsqueeze(-1)], dim=1) # concatenate class to end
	boxlist = boxes[boxes[:, 4] > conf_threshold].tolist() # filter by boxes with confidence > conf_threshold
	boxlist = sorted(boxlist, key=lambda x: x[4], reverse=True)

	good_boxes = []
	while len(boxlist) > 0:
		best_box = boxlist.pop(0)
		for box in boxlist:
			if int(best_box[5]) == int(box[5]):
				curr_iou = iou(torch.tensor(best_box[:4]).unsqueeze(0), torch.tensor(box[:4]).unsqueeze(0))
				if  curr_iou > iou_threshold:
					boxlist.remove(box)
		good_boxes.append(best_box)
	return torch.tensor(good_boxes)

@torch.no_grad()
def show_image(inp):
	fig, ax = plt.subplots()
	i, o = inp
	o = o[0]
	S = o.shape[0]
	i = i[0].permute(1, 2, 0).numpy()
	outs = nms(o, 0.50, 0.3)
	for o2 in outs:
		xmin = float(o2[0])
		ymin = float(o2[1])
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
	data = torchvision.datasets.VOCDetection("data", '2012', 'trainval', download=False, transform=transform)
	dataloader = DataLoader(YoloDataset(data, S, B, C, image_size), batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
	return dataloader
