# for final project

import numpy as np
import matplotlib.pyplot as plt

color = ['b', 'r--', 'c', 'y--', 'g:', 'k']

def plot_one(path, mode):
	with open(path, 'r') as f:
		lines = f.readlines()

	x = []
	loss = []
	acc = []
	test_acc = []
	for i in range(len(lines)):
		tmp = lines[i].split()
		if i%4==0:
			x.append(int(tmp[1]))
		elif i%4==1:
			loss.append(float(tmp[1]))
			acc.append(float(tmp[3]))
		elif i%4==2:
			test_acc.append(float(tmp[2]))


	x = np.array(x)
	#x = np.array([i for i in range(250)])

	if mode == 'loss':
		y = np.array(loss)
		plt.plot(x, y, color[0])
		plt.title('CNN6-mixup-in-higher-feature loss', fontsize=14)
		plt.xlabel('epoch', fontsize=12)
		plt.ylabel('loss', fontsize=12)
		plt.show()

	elif mode == 'acc':
		y = np.array(acc)
		plt.plot(x, y, color[0])

		y = np.array(test_acc)
		plt.plot(x, y, color[1])

		plt.title('CNN6-mixup-in-higher-feature accuracy', fontsize=14)
		plt.xlabel('epoch', fontsize=12)
		plt.ylabel('acc', fontsize=12)
		plt.legend(['train acc', 'test acc'])
		plt.show()

def plot_all(mode):
	lines = []
	'''
	with open('record_normal.txt', 'r') as f:
		lines1 = f.readlines()
		lines.append(lines1)
	with open('record_sd.txt', 'r') as f:
		lines2 = f.readlines()
		lines.append(lines2)
	with open('record_noisy_label.txt', 'r') as f:
		lines3 = f.readlines()
		lines.append(lines3)
	with open('record_gc.txt', 'r') as f:
		lines4 = f.readlines()
		lines.append(lines4)
	with open('record_lrd.txt', 'r') as f:
		lines5 = f.readlines()
		lines.append(lines5)
	with open('record_sd_lrd.txt', 'r') as f:
		lines6 = f.readlines()
		lines.append(lines6)
	'''
	with open('record_normal.txt', 'r') as f:
		lines4 = f.readlines()
		lines.append(lines4)

	with open('no-noise-cnn6-record.txt', 'r') as f:
		lines1 = f.readlines()
		lines.append(lines1)

	for i in range(len(lines)):
		x = []
		loss = []
		acc = []
		test_acc = []
		for j in range(len(lines[i])):
			tmp = lines[i][j].split()
			if j%4==0:
				x.append(int(tmp[1]))
			elif j%4==1:
				loss.append(float(tmp[1]))
				acc.append(float(tmp[3]))
			elif j%4==2:
				test_acc.append(float(tmp[2]))


		x = np.array(x)
		if mode == 'loss':
			y = np.array(loss)
		elif mode == 'train acc':
			y = np.array(acc)
		else:
			y = np.array(test_acc)

		plt.plot(x, y, color[i])

	plt.title('Training accuracy normal and no noise', fontsize=14)
	plt.xlabel('epoch', fontsize=12)
	plt.ylabel('acc', fontsize=12)
	#plt.legend(['WO', 'SD', 'NL', 'GC', 'LRD', 'SD_LRD'])
	plt.legend(['normal', 'no noise'])
	plt.show()
	

plot_all('train acc')


