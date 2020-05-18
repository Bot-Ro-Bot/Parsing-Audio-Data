import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

#module to output the sound 
from playsound import playsound

#metadata is a python file which contains a dictoinary of all the speakers and their detals
import metadata
"""
lets assume the users are tagged the following indices:
Speaker 0 : jackson
Speaker 1 : nicolas
Speaker 2 : theo
Speaker 3 : yweweler
"""

#length of all sample is 1 seconds
sample_length = 1   #in seconds
samples = [] ; sample_rate = []

#name of the folder containing the samples
dataset_folder = "recordings"
current_dir = os.listdir() 

os.chdir(current_dir[current_dir.index(dataset_folder)])
sample_dir =os.getcwd()
all_samples = os.listdir()


def extract_labels(all_samples):
	"""
	this function extracts the labels and speaker from the dataset
	"""
	labels = []
	speaker = []
	print("Extracting Labels")
	for i in range(len(all_samples)):
		temp = all_samples[i]
		temp = (temp[0:-3].split("_"))
		labels.append(temp[0])
		speaker.append(temp[1])
		if i%16==0:
			print("-",end="")
	print("\nLabels Extracted\n")
	return labels,speaker

def import_data(all_samples):
	"""
	this function imports all the wave files in dataset
	"""
	samples= []
	sample_rate = []
	print("Loading Dataset")
	for i in range(len(all_samples)):
		s_len, s = wavfile.read(all_samples[i])
		samples.append(s);	sample_rate.append(s_len)
		if i%16==0:
			print("-",end="")
	print("\nDataset Loaded\n")
	return samples,sample_rate


samples,sample_rate = import_data(all_samples)
labels,speaker= extract_labels(all_samples)
print(len(samples),len(sample_rate),len(labels),len(speaker))
print(type(samples))
data = pd.DataFrame(labels,speaker)

print(data.head(10))

