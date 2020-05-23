import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fftpack
from scipy import signal
from scipy.stats import zscore
import os
from sklearn import svm
import tensorflow as tf 
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier



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

def ZCR(sample):
	sample = sample - sample.mean()
	pos = sample>0
	npos = ~pos
	return len(((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0])

def RMS(sample):
	return np.sqrt(np.mean(np.power(sample,2)))

def temporal_features(sample,frames):
	"""
	temporal features are the features in time domain
	some of them are:
	1. Standard deviation
	2. Min value , Max Value
	3. Root mean Square
	4. Mean """
	features = []
	for i in range(frames):
		temp = sample[int(frames*i):int(frames*(i+1))]
		feat = [np.mean(temp), np.std(temp), np.min(temp),np.max(temp),RMS(temp)]
		features.append(feat)
	return np.nan_to_num(np.hstack(features))

# def train_test_split(samples , labels):
# 	#do a 75:25 split of data into training and testing data
# 	margin = int(0.80*len(samples))
# 	train_data = samples[:margin] 
# 	train_label = labels[:margin]
# 	test_data = samples[margin:]
# 	test_label = labels[margin:]
# 	return train_data,train_label,test_data,test_label


def extract_features(samples,frames):
	features = []
	print("Extracting features")
	i = 0
	for sample in samples:
		i+=1
		feat = temporal_features(sample,frames)
		features.append(feat)
		if i%16==0:
			print("-",end="")
	print("\nFeatures Extracted\n")
	return np.vstack(features)

samples,sample_rate = import_data(all_samples)
#since every sample is sampled at the same sample rate , we can take only one value tp represent every sample
sample_rate = sample_rate[0]
print("sample_rate",sample_rate)
labels, speakers = extract_labels(all_samples)

# samples = zero_padding(samples)
samples = zero_padding(samples)
samples = normalize_data(samples)
print(len(samples[0]))

#size of each window taken for temporal analysis
"""
In this work we take in consideration a window size that is 1/5 of the maximum duration of 200 ms
to consider also short consonant and vowels. For these reasons the window size has been selected
to be equal to 40 ms and the number of the corresponding samples has been taken into account
accordingly to the sampling rate.
(cited from a pdf linked in the same repo)
"""
window_size = 0.020 #in seconds
frames = int( (sample_rate /(2*len(samples[0]))) / window_size)
print(frames)

features = extract_features(samples,frames)
print((features.shape))


train_data , test_data , train_label, test_label = train_test_split(features, labels,test_size=0.10)
# train_data , train_label, test_data ,test_label = train_test_split(features, labels)
print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

# train_data , train_label , test_data , test_label= train_data.tolist() , train_label.tolist() , test_data.tolist(), test_label.tolist()
# print(type(train_data))
# print(type(train_data[0]))
print(type(test_data[50]))
print(len(test_data[50]))
# print(type(train_label[0]))
# print(type(test_label[0]))


model = keras.Sequential()
model.add(keras.layers.Dense(288 , activation= 'relu', input_shape=(train_data.shape[1],)))
# model.add(keras.layers.Flatten()),
model.add(keras.layers.Dense(100 , activation = 'relu'))
model.add(keras.layers.Dense(100 , activation = 'relu'))
model.add(keras.layers.Dense(64 , activation = 'relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data,train_label,epochs=20,batch_size=100)

test_loss, test_accuracy = model.evaluate(test_data,test_label)
print("Tested accuracy: ", test_accuracy)
print("Tested loss: ", test_loss)

pred = model.predict(test_data)
# print((prediction[50]))
# print("prediction should be: " , test_label[50])
points = 0 

for i in range(len(pred)):
	print(int(np.argmax(pred[i])),int(test_label[i]))
	if ( int(np.argmax(pred[i])) == int(test_label[i]) ):
		points+=1
print("Accuracy : ", ( (points/ len(pred)) * 100))