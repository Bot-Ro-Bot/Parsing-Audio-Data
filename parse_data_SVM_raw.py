import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fftpack
from scipy import signal
from scipy.stats import zscore
import os
from sklearn import svm

# import soundfile as sf
#import pyAudioAnalysis
#module to output the sound 
from playsound import playsound

#metadata is a python file which contains a dictoinary of all the speakers and their detals
# import metadata
"""
lets assume the users are tagged the following indices:
Speaker 0 : jackson
Speaker 1 : nicolas
Speaker 2 : theo
Speaker 3 : yweweler
"""

#length of all sample is 1 seconds approx
sample_length = 1   #in seconds
samples = [] ; sample_rate = []

#name of the folder containing the samples
dataset_folder = "recordings"
current_dir = os.listdir() 
main_dir = os.getcwd()
os.chdir(current_dir[current_dir.index(dataset_folder)])
sample_dir = os.getcwd()
all_samples = os.listdir()

# all_samples.sort()
print((all_samples[2]))

def extract_labels(all_samples):
	"""
	this function extracts the labels and speaker from the dataset
	"""
	labels = []
	speakers = []
	print("Extracting Labels")
	for i in range(len(all_samples)):
		temp = all_samples[i]
		temp = (temp[0:-3].split("_"))
		labels.append(float(temp[0]))
		speakers.append(temp[1])
		if i%16==0:
			print("-",end="")
	print("\nLabels Extracted\n")
	return np.array(labels),np.array(speakers)

def import_data(all_samples):
	"""
	this function imports all the wave files in dataset
	"""
	samples = []
	sample_rate = []
	print("Loading Dataset")
	for i in range(len(all_samples)):
		s_len, s = wavfile.read(all_samples[i])
		samples.append(s);	
		sample_rate.append(s_len)
		if i%16==0:
			print("-",end="")
	print("\nDataset Loaded\n")
	return np.array(samples),sample_rate

def normalize_data(samples):
	return [zscore(sample) for sample in samples]

def zero_padding(samples):
	"""
	this function pads the samples to make every sample the equal length
	it cuts off excess values and addes zeros to the insufficient ones
	making the length a power of 2 makes the calculation of fft faster and convinient
	"""
	for i in range(len(samples)):
		length = len(samples[i])
		diff = int(abs(4096-length) / 2)
		diff = abs(4096-length)
		pad0 = int(diff/2)
		pad1 = diff-pad0	
		if(length == 4096):
			continue
		elif(length < 4096):
			samples[i] = np.pad(samples[i],(pad0,pad1))
		else:
			#chopping the signals with  higher number of datas
			samples[i] = samples[i][pad0:-pad1]
	return samples


def train_test_split(samples , labels):
	#do a 75:25 split of data into training and testing data
	margin = int(0.75*len(samples))
	train_data = samples[:margin] 
	train_label = labels[:margin]
	test_data = samples[margin:]
	test_label = labels[margin:]
	return train_data,train_label,test_data,test_label


samples,sample_rate = import_data(all_samples)
# print(samples)
labels, speakers = extract_labels(all_samples)

# samples = zero_padding(samples)
samples = zero_padding(samples)
samples = normalize_data(samples)
# print(samples)

train_data , train_label, test_data ,test_label = train_test_split(samples, labels)
# print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

# print(train_data.flatten())
clf = svm.SVC()
clf.fit(train_data,train_label)
# pred = (clf.predict([test_data[10]]))
pred = (clf.predict(test_data))
# print(pred)
points = 0 

for i in range(len(pred)):
	print(int(pred[i]),test_label[i])
	if ( int(pred[i]) == int(test_label[i]) ):
		points+=1
print("Accuracy : ", ( (points/ len(pred)) * 100))

# print("Value should be :", test_label[10])