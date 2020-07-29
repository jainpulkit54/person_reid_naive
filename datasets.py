import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms

class ImageFolder(data.Dataset):

	def __init__(self, folder_path):
		self.folder_path = folder_path
		self.targets_name = sorted(os.listdir(self.folder_path))
		subfolder_name = []
		images_name = []
		targets = []
		
		for folder_name in self.targets_name:
			path = os.listdir(self.folder_path + folder_name + '/')
			n_images = len(path)
			images_name.extend(path)
			subfolder_name.extend([(folder_name + '/')]*n_images)
			targets.extend([int(folder_name)]*n_images)
		
		self.subfolder_name = subfolder_name
		self.images_name = images_name
		self.targets = targets
		self.total_samples = len(self.targets)
		self.totensor = transforms.ToTensor()
		self.horizontal_flip = transforms.RandomHorizontalFlip(p = 0.5)

	def __getitem__(self, index):
		
		img = Image.open(self.folder_path + self.subfolder_name[index] + self.images_name[index]).convert('RGB')
		img = self.horizontal_flip(img)
		img = self.totensor(img)
		target = self.targets[index]
		
		return img, target

	def __len__(self):
		
		return self.total_samples

class Triplet_Generation(data.Dataset):

	def __init__(self, dataset):
		
		self.data = dataset
		self.train_data = []
		for i in range(len(dataset)):
			self.train_data.append(self.data[i][0])
		self.train_targets = np.array(dataset.targets)
		self.labels_set = set(self.train_targets)
		self.labels_to_indices = {label: np.where(self.train_targets == label)[0] for label in self.labels_set}

	def __getitem__(self, index):
		
		anchor_img = self.train_data[index]
		anchor_class = self.train_targets[index].item()
		ind = np.random.choice(self.labels_to_indices[anchor_class],1)[0]
		postive_img = self.train_data[ind]
		new_set = self.labels_set - set([anchor_class])
		negative_label = np.random.choice(list(new_set), 1)
		ind = np.random.choice(self.labels_to_indices[negative_label[0]],1)[0]
		negative_img = self.train_data[ind]
		img1 = anchor_img
		img2 = postive_img
		img3 = negative_img

		return img1, img2, img3

	def __len__(self):

		return int(len(self.train_data))