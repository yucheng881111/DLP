from dataloader import RetinopathyLoader
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, grad):
	if grad:
		for param in model.parameters():
			param.requires_grad = True
	else:
		for param in model.parameters():
			param.requires_grad = False


def get_model1():
	resnet18 = models.resnet18(pretrained = True)
	set_parameter_requires_grad(resnet18, False)
	num_ftrs = resnet18.fc.in_features
	resnet18.fc = nn.Linear(num_ftrs, 5)
	return resnet18

def get_model2():
	resnet50 = models.resnet50(pretrained = True)
	set_parameter_requires_grad(resnet50, False)
	num_ftrs = resnet50.fc.in_features
	resnet50.fc = nn.Linear(num_ftrs, 5)
	return resnet50


model = get_model2()
model.to(device)

train_data = RetinopathyLoader(root = 'data/', mode = 'train')
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = 16, shuffle = True)

test_data = RetinopathyLoader(root = 'data/', mode = 'test')
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = 16, shuffle = True)

optimizer_feature = torch.optim.SGD(model.fc.parameters(), lr = 1e-3, momentum = 0.9)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 5e-4)

'''
nSamples = [20656, 1955, 4210, 698, 581]
normedWeights = [1 - (x / 28100) for x in nSamples]
baseline = normedWeights[0]
normedWeights = [x / baseline for x in normedWeights]
print('each class weights: ' + str(normedWeights))
normedWeights = torch.FloatTensor(normedWeights).to(device)
'''
#loss_func = nn.CrossEntropyLoss(weight = normedWeights)
loss_func = nn.CrossEntropyLoss()
num_epochs = 20

total_train_accuracy = []
total_test_accuracy = []
max_acc = -1
max_p = []
max_l = []
max_e = -1
for epoch in range(num_epochs):
	# training
	model.train()
	correct_train = 0
	total_train = 0
	total_loss = 0
	feature_extraction = False
	if epoch < 5:
		print('feature_extracting...')
		feature_extraction = True
	else:
		set_parameter_requires_grad(model, True)

	for i, (data, labels) in enumerate(train_loader):
		data = data.to(device, dtype = torch.float)
		labels = labels.to(device, dtype = torch.long)

        # Clear gradients
		if feature_extraction:
			optimizer_feature.zero_grad()
		else:
			optimizer.zero_grad()

        # Forward
		outputs = model(data)

        # Calculate cross entropy loss
		train_loss = loss_func(outputs, labels)
		total_loss += float(train_loss)

        # Get predictions from the maximum value
		predicted = torch.max(outputs.data, 1)[1]
        
        # Total number of labels
		total_train += len(labels)

        # Total correct predictions
		correct_train += (predicted == labels).float().sum()

        # Calculate gradients
		train_loss.backward()

        # Update parameters
		if feature_extraction:
			optimizer_feature.step()
		else:
			optimizer.step()

	train_accuracy = 100 * (correct_train / total_train)
	total_train_accuracy.append(train_accuracy.item())

	
	print('\nepoch ' + str(epoch + 1) + ':')
	print('trainig accuracy: ' + str(train_accuracy) + '  loss: ' + str(total_loss))

	#testing
	model.eval()
	total_test = 0
	correct_test = 0
	pred = []
	lab = []
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
		
		for p in predicted.cpu().tolist():
			pred.append(p)
		for l in labels.cpu().tolist():
			lab.append(l)

	test_accuracy = 100 * (correct_test / total_test)
	if test_accuracy.item() > max_acc:
		max_acc = test_accuracy.item()
		max_p = pred
		max_l = lab
		max_e = epoch

	total_test_accuracy.append(test_accuracy.item())

	print('testing accuracy: ' + str(test_accuracy))


print('')
print('max testing accuracy ' + str(max_acc) + '% at epoch ' + str(max_e + 1))

with open('pred.txt', 'w') as f:
	for i in max_p:
		f.write(str(i) + ' ')

with open('lab.txt', 'w') as f:
	for i in max_l:
		f.write(str(i) + ' ')

with open('train_acc.txt', 'w') as f:
	for i in total_train_accuracy:
		f.write(str(i) + '\n')

with open('test_acc.txt', 'w') as f:
	for i in total_test_accuracy:
		f.write(str(i) + '\n')

# save model
torch.save(model.state_dict(), 'model_resnet50.pt')

