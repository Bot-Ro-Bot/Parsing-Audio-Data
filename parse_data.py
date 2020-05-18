import pandas as pd 
import numpy as np
# import librosa 
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fftpack
from scipy import signal
import os

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

#length of all sample is 1 seconds
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
# print((all_samples[2]))



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
		labels.append(temp[0])
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

def extract_features(sample):
	"""
	this function extracts the features from the input dataset , 
	only input one sample at a time
	only one sample/input is taken to minimize the vast numbers of output to a few numpy n dimensional array 
	1) Compactness.
	2) Magnitude spectrum.
	3) Mel-frequency cepstral coefficients.
	4) Pitch.
	5) Power Spectrum.
	6) RMS.
	7) Rhythm.
	8) Spectral Centroid.
	9) Spectral Flux.
	10) Spectral RollOff Point.
	11) Spectral Variability.
	12) Zero Crossings
	"""

	pass

def visualize_data(sample,mutiple="False"):
	"""
	one or mutiple sample can be given as a input to this function
	makes the figures accordingy
	set mutiple to true if multiple samples are passed
	"""

	for i in range(sample.shape[0]):
		plt.figure(i)
		plt.subplot(411)

	pass

def zero_padding(samples):
	"""
	this function pads the samples to make every sample the equal length
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
			samples[i] = samples[i][pad0:-pad1]
	return samples

# def make_dataframe():
# 	pass

samples,sample_rate = import_data(all_samples)
labels, speakers = extract_labels(all_samples)
print(len(samples[1000]),len(sample_rate),len(labels),len(speakers))

plt.subplot(311)
plt.plot(samples[1000])

# samples = zero_padding(samples)
samples = zero_padding(samples)
print(len(samples[1000]))


#putting the parsed data into a dataframe
data = np.transpose(np.array([samples ,sample_rate,labels,speakers]))
data = pd.DataFrame(data,
	columns=['Audio_Data','Sample_Rate','Labels', 'Speakers'])

#exploring the formed data frame
# print(data.head(10))

#saving the dataframe into a file (excel file, html table file )
# data.to_excel(main_dir+"/parsed_data.xlsx")
# data.to_html(main_dir+"/parsed_data.html")

# window = signal.blackman(len(samples[0]))
# p, p_gram, gram = signal.spectrogram(samples[0],fs=8000)
# # s_gram = 
# rfft = fftpack.fft((samples[0]*window),n=4000)
# irfft = fftpack.ifft(rfft)
# plt.subplot(311)
# plt.plot(samples[0])

# plt.subplot(312)
# plt.plot(samples[0])

# # plt.plot(abs(rfft[0:int (len(rfft)/2)]))
# plt.specgram(samples[0],Fs=8000)
# plt.subplot(313)
# # plt.semilogy(p,p_gram)
# # plt.ylim([1e-7, 1e2])
# plt.pcolormesh(p_gram,p,gram)
plt.show()