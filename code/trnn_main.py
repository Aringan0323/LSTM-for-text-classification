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
def model_train(all_data, word_embed, train_data_x, train_data_y, test_data_x, test_data_y, epochs, device):
	train_data_x = torch.LongTensor(train_data_x)
	train_data_y = torch.LongTensor(train_data_y)
	test_data_x = torch.LongTensor(test_data_x)
	test_data_y = torch.LongTensor(test_data_y)
	train_set = TensorDataset(train_data_x, train_data_y)
	trainloader = DataLoader(train_set, batch_size=5, shuffle=True, num_workers=3)
	test_set = TensorDataset(test_data_x, test_data_y)
	testloader = DataLoader(test_set, batch_size=5, shuffle=True, num_workers=3)
	word_embed = torch.from_numpy(word_embed).to(device)
	p_content = torch.FloatTensor(all_data).to(device)

	model = U.Text_Encoder(p_content, word_embed, mean_pooling=False, device=device)
	model = model.to(device)
	model_loss = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	for epoch in range(epochs):
		running_loss = 0.0
		for i, data in enumerate(trainloader,0):
			inputs, labels = data[0], data[1]
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()


			outputs = model(inputs)
			outputs = torch.transpose(outputs, 0, 1)
			# if i % 1000 == 0:
			# 	print("Batch {} inputs shape: {}".format(i, p_content[inputs[0]]))
			# 	print("Batch {} labels shape: {}".format(i, labels.shape))
			# 	print("Batch {} outputs shape: {}".format(i, outputs.shape))
			loss = model_loss(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		loss_this_epoch = running_loss / len(train_data_x)
		print("Epoch {} loss: {}".format(epoch, loss_this_epoch))

# model test: can be called directly in model_train
def model_test(test_data_x, test_data_y, net, epoch_num):
	return


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
	model_train(all_data, word_embed, train_data_x, train_data_y, test_data_x, test_data_y, 50, torch.device('cpu'))
