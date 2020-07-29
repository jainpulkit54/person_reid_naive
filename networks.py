import torch.nn as nn
from torchvision import models

class EmbeddingNet(nn.Module):

	def __init__(self):
		super(EmbeddingNet, self).__init__()
		resnet50 = models.resnet50(pretrained = True, progress = True)
		resnet50 = list(resnet50.children())[:-1]
		self.conv_layers = nn.Sequential(*resnet50)
		self.fc_layers = nn.Sequential(
			nn.Linear(2048, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 128)
			)

	def forward(self, x1, x2, x3):
		x1 = self.conv_layers(x1)
		x1 = x1.view(x1.shape[0],-1)
		x1 = self.fc_layers(x1)

		x2 = self.conv_layers(x2)
		x2 = x2.view(x2.shape[0],-1)
		x2 = self.fc_layers(x2)

		x3 = self.conv_layers(x3)
		x3 = x3.view(x3.shape[0],-1)
		x3 = self.fc_layers(x3)

		return x1, x2, x3

	def get_embeddings(self, x1, x2, x3):
		return self.forward(x1, x2, x3)

class EmbeddingNet_2(nn.Module):

	def __init__(self):
		super(EmbeddingNet_2, self).__init__()
		resnet50 = models.resnet50(pretrained = True, progress = True)
		resnet50 = list(resnet50.children())[:-1]
		self.conv_layers = nn.Sequential(*resnet50)
		self.fc_layers = nn.Sequential(
			nn.Linear(2048, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 128)
			)

	def forward(self, x):
		x = self.conv_layers(x)
		x = x.view(x.shape[0],-1)
		x = self.fc_layers(x)
		return x

	def get_embeddings(self, x):
		return self.forward(x)