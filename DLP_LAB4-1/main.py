import torch
import torch.nn as nn
import numpy
from dataloader import read_bci_data
from EEG import EEGNet
from DeepConv import DeepConvNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: ' + str(device))
print('')

X_train, y_train, X_test, y_test = read_bci_data()
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

train = torch.utils.data.TensorDataset(X_train, y_train)
test = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size = 256, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = 256, shuffle = True)

activate_func = [nn.ELU(), nn.ReLU(), nn.LeakyReLU()]
ELU_max = 0
ReLU_max = 0
LeakyReLU_max = 0
cnt = 0
for a in activate_func:
	func = ''
	if cnt == 0:
		func = 'ELU'
	elif cnt == 1:
		func = 'ReLU'
	else:
		func = 'LeakyReLU'
	print('\nUsing ' + func + '...')

	model = EEGNet(a, device)
	#model = DeepConvNet(a, device)
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.01)
	loss_func = nn.CrossEntropyLoss()
	num_epochs = 500

	total_train_accuracy = []
	total_test_accuracy = []
	for epoch in range(num_epochs):
		# training
		correct_train = 0
		total_train = 0
		total_loss = 0
		for i, (data, labels) in enumerate(train_loader):
			data = data.to(device, dtype = torch.float)
			labels = labels.to(device, dtype = torch.long)

	        # Clear gradients
			optimizer.zero_grad()

	        # Forward
			outputs = model(data)

	        # Calculate cross entropy loss
			train_loss = loss_func(outputs, labels)
			total_loss += train_loss

	        # Get predictions from the maximum value
			predicted = torch.max(outputs.data, 1)[1]
	        
	        # Total number of labels
			total_train += len(labels)

	        # Total correct predictions
			correct_train += (predicted == labels).float().sum()

	        # Calculate gradients
			train_loss.backward()

	        # Update parameters
			optimizer.step()

		train_accuracy = 100 * (correct_train / total_train)
		total_train_accuracy.append(train_accuracy.item())

		if epoch % 10 == 9:
			print('\nepoch ' + str(epoch + 1) + ':')
			print('trainig accuracy: ' + str(train_accuracy) + '  loss: ' + str(total_loss))

		#testing
		total_test = 0
		correct_test = 0
		for i, (data, labels) in enumerate(test_loader):
			data = data.to(device, dtype = torch.float)
			labels = labels.to(device, dtype = torch.long)

			# Forward
			outputs = model(data)

			# Get predictions from the maximum value
			predicted = torch.max(outputs.data, 1)[1]
	        
	        # Total number of labels
			total_test += len(labels)

	        # Total correct predictions
			correct_test += (predicted == labels).float().sum()

		test_accuracy = 100 * (correct_test / total_test)
		total_test_accuracy.append(test_accuracy.item())

		if epoch % 10 == 9:
			print('testing accuracy: ' + str(test_accuracy))

	max_acc = max(total_test_accuracy)
	print('')
	print(func + ' has max accuracy ' + str(max_acc) + '% at epoch ' + str(total_test_accuracy.index(max_acc)))
	if cnt == 0:
		ELU_max = max_acc
	elif cnt == 1:
		ReLU_max = max_acc
	else:
		LeakyReLU_max = max_acc
	cnt += 1

	#with open('Deep_train_' + str(cnt) + '.txt', 'w') as f:
	#	for i in total_train_accuracy:
	#		f.write(str(i) + '\n')
	#with open('Deep_test_' + str(cnt) + '.txt', 'w') as f:
	#	for i in total_test_accuracy:
	#		f.write(str(i) + '\n')



print('')
print('ELU has max accuracy ' + str(ELU_max) + '%')
print('ReLU has max accuracy ' + str(ReLU_max) + '%')
print('LeakyReLU has max accuracy ' + str(LeakyReLU_max) + '%')




