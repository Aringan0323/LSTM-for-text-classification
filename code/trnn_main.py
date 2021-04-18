import numpy as np
# import matplotlib.pyplot as plt
# import scipy
import trnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(0)

# model train
def model_train(all_data, word_embed, train_data_x, train_data_y, test_data_x, test_data_y):
	


# model test: can be called directly in model_train 
def model_test(test_data_x, test_data_y, net, epoch_num):
	


if __name__ == '__main__':
	# load datasets
	input_data = U.input_data()

	all_data, train_data, test_data = input_data.load_text_data()
	train_data_x = train_data[0] # map content by id
	train_data_y = train_data[1]
	test_data_x = test_data[0] # map content by id
	test_data_y = test_data[1]

	word_embed = input_data.load_word_embed()

	# model train (model test function can be called directly in model_train)
	model_train(all_data, word_embed, train_data_x, train_data_y, test_data_x, test_data_y)






