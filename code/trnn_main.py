import numpy as np
import sys
import matplotlib.pyplot as plt
# import scipy
import trnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(0)

# model train
def model_train(all_data, word_embed, train_data_x, train_data_y, test_data_x, test_data_y, epochs, device, mean_pooling=False):
	num_batches = len(train_data_x)/5
	train_data_x = torch.LongTensor(train_data_x)
	train_data_y = torch.LongTensor(train_data_y)
	test_data_x = torch.LongTensor(test_data_x)
	test_data_y = torch.LongTensor(test_data_y)
	train_set = TensorDataset(train_data_x, train_data_y)
	trainloader = DataLoader(train_set, batch_size=5, shuffle=True, num_workers=10)
	word_embed = torch.from_numpy(word_embed).to(device)
	p_content = torch.FloatTensor(all_data).to(device)

	model = U.Text_Encoder(p_content, word_embed, mean_pooling=mean_pooling, device=device)
	model = model.to(device)
	model_loss = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	history = {'epoch':[], 'loss':[], 'acc':[]}
	for epoch in range(epochs):
		running_loss = 0.0
		for i, data in enumerate(trainloader,0):
			portion_done = i/num_batches
			sys.stdout.write("\033[K") #clear line

			print("\tEpoch {}: [".format(epoch) + "#" * int(portion_done*50) + "-" * int((1-portion_done)*50) + "] {}%".format(round(portion_done*100, 1)), end='\r')
			inputs, labels = data[0], data[1]
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()


			outputs = model(inputs)
			loss = model_loss(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		loss_this_epoch = running_loss / len(train_data_x)
		accuracy = model_test(test_data_x, test_data_y, model, epoch, device)
		history['epoch'].append(epoch)
		history['loss'].append(loss_this_epoch)
		history['acc'].append(accuracy)
		print("Epoch {} \n   loss: {}\n   accuracy: {}".format(epoch, loss_this_epoch, accuracy))
	return history

# model test: can be called directly in model_train
def model_test(test_data_x, test_data_y, model, epoch_num, device):
	test_set = TensorDataset(test_data_x, test_data_y)
	testloader = DataLoader(test_set, batch_size=5, shuffle=True, num_workers=10)
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			inputs, labels = data[0].to(device), data[1].to(device)
			outputs = model(inputs)
			total += labels.shape[0]

			predicted = torch.argmax(outputs, dim=1)
			correct += torch.sum(predicted == labels)
	accuracy = (100*correct/total)
	return accuracy


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
	hist1 = model_train(all_data, word_embed, train_data_x, train_data_y, test_data_x, test_data_y, 10, torch.device('cpu'))
	hist2 = model_train(all_data, word_embed, train_data_x, train_data_y, test_data_x, test_data_y, 10, torch.device('cpu'), mean_pooling=True)


	plt.plot(hist1['epoch'], hist1['acc'], label='Last Hidden Layer Encoding')
	plt.plot(hist2['epoch'], hist2['acc'], label='Mean Pooling Encoding')
	plt.axis([0,10,0,100])
	plt.legend()
	plt.xlabel('Epochs')
	plt.ylabel('Test Accuracy (Percent)')
	plt.title('Accuracy over Epochs of LSTM Text Classifier')

	plt.savefig('../results/LSTM_Results.png')
	plt.close()
