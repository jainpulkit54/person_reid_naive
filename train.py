import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from networks import *
from loss_functions import *
from datasets import *

writer = SummaryWriter('logs_market1501_naive/person_reid')
os.makedirs('checkpoints_market1501_naive', exist_ok = True)

batch_size = 128
train_dataset = ImageFolder(folder_path = '../market_1501/')
train_triplet_dataset = Triplet_Generation(train_dataset)
train_loader = DataLoader(train_triplet_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
no_of_training_batches = len(train_loader)/batch_size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 50

embeddingNetTriplet = EmbeddingNet()
optimizer = optim.Adam(embeddingNetTriplet.parameters(), lr = 3e-5, betas = (0.9, 0.999))

def run_epoch(data_loader, model, optimizer, epoch_count = 0):

	model.to(device)
	model.train()
	
	running_loss = 0.0

	for batch_id, (anchor_imgs, positive_imgs, negative_imgs) in enumerate(data_loader):
		
		iter_count = epoch_count * len(data_loader) + batch_id
		anchor_imgs = anchor_imgs.to(device)
		positive_imgs = positive_imgs.to(device)
		negative_imgs = negative_imgs.to(device)
		emb_anchor, emb_postive, emb_negative = model.get_embeddings(anchor_imgs, positive_imgs, negative_imgs)
		embeddings_2_norm_anchor = torch.mean(torch.norm(emb_anchor, p = 2, dim = 1))
		embeddings_2_norm_positive = torch.mean(torch.norm(emb_postive, p = 2, dim = 1))
		embeddings_2_norm_negative = torch.mean(torch.norm(emb_negative, p = 2, dim = 1))
		batch_loss, non_zero_losses = triplet_loss(emb_anchor, emb_postive, emb_negative, batch_size, margin = 0.2)
		optimizer.zero_grad()
		
		batch_loss.backward()
		optimizer.step()

		running_loss = running_loss + batch_loss.item()
		writer.add_scalar('Batch_Wise_Triplet_Loss', batch_loss.item(), iter_count)
		writer.add_scalar('2-norm of Embeddings Anchor', embeddings_2_norm_anchor ,iter_count)
		writer.add_scalar('2-norm of Embeddings Positive', embeddings_2_norm_positive ,iter_count)
		writer.add_scalar('2-norm of Embeddings Negative', embeddings_2_norm_negative ,iter_count)
		writer.add_scalar('% non-zero losses in batch', (non_zero_losses*100), iter_count)

	return running_loss

def fit(data_loader, model, optimizer, n_epochs):

	print('Training Started\n')
	
	for epoch in range(n_epochs):
		
		loss = run_epoch(data_loader, model, optimizer, epoch_count = epoch)
		loss = loss/no_of_training_batches

		print('Loss after epoch ' + str(epoch + 1) + ' is:', loss)
		torch.save({'state_dict': model.cpu().state_dict()}, 'checkpoints_market1501_naive/model_epoch_' + str(epoch + 1) + '.pth')

fit(train_loader, embeddingNetTriplet, optimizer = optimizer, n_epochs = epochs)