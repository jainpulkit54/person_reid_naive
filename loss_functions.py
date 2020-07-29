import torch
import torch.nn as nn

def triplet_loss(emb_anchor, emb_positive, emb_negative, batch_size, margin = 0.2):
	
	max_fn = nn.ReLU()
	positive_pair_loss = torch.sum((emb_anchor - emb_positive)**2, dim = 1)
	negative_pair_loss = torch.sum((emb_anchor - emb_negative)**2, dim = 1)
	triplet_loss = max_fn(margin + positive_pair_loss - negative_pair_loss)
	
	# Counting the number of positive triplets (i.e., places where loss > 0)
	positive_loss_triplets = torch.ge(triplet_loss, 1e-16).float() # i.e., 0 will be 1e-16 or a very small value
	num_positive_loss_triplets = torch.sum(positive_loss_triplets)
	fraction_positive_loss_triplets = num_positive_loss_triplets / batch_size
	triplet_loss = torch.sum(triplet_loss)
	
	return triplet_loss, fraction_positive_loss_triplets