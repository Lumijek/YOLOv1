import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bn=True, alpha=0.1):
		super().__init__()
		if padding == None:
			padding = kernel_size // 2

		norm = nn.Identity()
		if bn == True:
			norm = nn.BatchNorm2d(out_channels)

		self.layer = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
			norm,
			nn.LeakyReLU(alpha)
		)

	def forward(self, x):
		return self.layer(x)

class YOLOv1(nn.Module):
	def __init__(self, S=7, B=2, C=20):
		super().__init__()

		self.S = S
		self.B = B
		self.C = C
		# ConvNet
		self.conv1 = ConvLayer(3, 64, 7, stride=2)
		self.mp1 = nn.MaxPool2d(2)

		self.conv2 = ConvLayer(64, 192, 3)
		self.mp2 = nn.MaxPool2d(2)

		self.conv3 = nn.Sequential(
			ConvLayer(192, 128, 1),
			ConvLayer(128, 256, 3),
			ConvLayer(256, 256, 1),
			ConvLayer(256, 512, 3)
		)
		self.mp3 = nn.MaxPool2d(2)

		conv4_block = []
		for _ in range(4):
			conv4_block.append(ConvLayer(512, 256, 1))
			conv4_block.append(ConvLayer(256, 512, 3))

		conv4_block.append(ConvLayer(512, 512, 1))
		conv4_block.append(ConvLayer(512, 1024, 3))

		self.conv4 = nn.Sequential(*conv4_block)
		self.mp4 = nn.MaxPool2d(2)

		self.conv5 = nn.Sequential(
			ConvLayer(1024, 512, 1),
			ConvLayer(512, 1024, 3),
			ConvLayer(1024, 512, 1),
			ConvLayer(512, 1024, 3),
			ConvLayer(1024, 1024, 3),
			ConvLayer(1024, 1024, 3, stride=2)
		)

		self.conv6 = nn.Sequential(
			ConvLayer(1024, 1024, 3),
			ConvLayer(1024, 1024, 3)
		)

		# FC

		self.ffn = nn.Sequential(
			nn.Flatten(),
			nn.Linear(1024 * 7 * 7, 4096),
			nn.BatchNorm1d(4096),
			nn.LeakyReLU(0.1),
			nn.Linear(4096, S * S * (B * 5 + C)),
			nn.Sigmoid(),
		)

	def forward(self, x):
		x = self.mp1(self.conv1(x))
		x = self.mp2(self.conv2(x))
		x = self.mp3(self.conv3(x))
		x = self.mp4(self.conv4(x))
		x = self.conv5(x)
		x = self.conv6(x)
		x = self.ffn(x)
		x = x.view(-1, self.S, self.S, (self.B * 5 + self.C))
		return x

