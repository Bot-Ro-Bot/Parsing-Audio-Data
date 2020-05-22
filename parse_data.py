import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fftpack
from scipy import signal
import os

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
print((all_samples[2]))


class Feature_Extraction:
	"""
	this class extracts the features from the input dataset ,  
	A list of probable features are:
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

	def __init__(self,sample,mutiple=False):
		self.sample = sample 
		self.mutiple = False

	def ZCR(self):
		ZCR = []
		for i in range(len(self.sample)):
			#for zero crossing it is necessary that the data points are centered around a mean values 
			# self.sample[i] = self.sample[i] - self.sample[i].mean()
			pos = self.sample[i]>0
			npos = ~pos
			ZCR.append(len(((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]))
		return np.array(ZCR)

		#the for loop for zcr calculation takes about 342 milliseconds more 
		#for a single input value  to compute than the stackflow one
		# ZCR = 0 
		# for i in range(1,len(sample)):
		# 	prev_sign = np.signbit(sample[i-1])
		# 	cur_sign = np.signbit(sample[i])
		# 	if( (cur_sign != prev_sign)  ):
		# 		ZCR+=1


	def spect(self):
		freq =[]
		time = []
		power = []

		for i in range(len(self.sample)):
			f, t, p = signal.spectrogram(self.sample[i],fs=8000,window='blackman')  #works good
			freq.append(list(f))
			time.append(t)
			power.append(p)
		return np.array(freq),np.array(time), np.array(power)

	def plot_spectogram(self):
		# select a random file from the dataset
		random = np.random.randint(0,2000)
		freq , t , power = signal.spectrogram(self.sample[random],fs=8000,window='blackman')
		plt.figure(1)
		plt.subplot(311)
		plt.plot(self.sample[random])

		plt.subplot(312)
		plt.specgram(self.sample[random],Fs=8000)
		plt.colorbar()

		plt.subplot(313)
		plt.pcolormesh(t,freq,power)
		pass

	def print_ZCR(self):
		print(self.ZCR())



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

def normalize_data(samples):
	print("Normalizing Data ")
	for i in range(len(samples)):
		samples[i] = ( samples[i] - samples[i].mean() ) / samples[i].std()
		if i%16==0:
			print("-",end="")
	print("\nData Normalized\n")
	return samples

def extract_features(sample):
	# sample = np.array(sample)
	
	
	sample = sample - sample.mean()
	# sample = sample[sample!=0]
	#code from stackflow
	# pos = sample>0
	# npos = ~pos
	# return len(((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0])
	# return ZCR
	return signal.spectrogram(sample,fs=8000,window='blackman')  #works good

	# pass

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

def make_dataframe(all_samples):
	"""
	
	"""
	samples,sample_rate = import_data(all_samples)
	labels, speakers = extract_labels(all_samples)
	# samples = zero_padding(samples)
	samples = zero_padding(samples)


	#putting the parsed data into a dataframe
	data = np.transpose(np.array([samples ,sample_rate,labels,speakers]))
	data = pd.DataFrame(data,
		columns=['Audio_Data','Sample_Rate','Labels', 'Speakers'])
	
	#exploring the formed data frame
	# print(data.head(10))

	#saving the dataframe into a file (excel file, html table file )
	# data.to_excel(main_dir+"/parsed_data.xlsx")
	# data.to_html(main_dir+"/parsed_data.html")

	return data

def train_test_split(samples):
	pass


samples,sample_rate = import_data(all_samples)
labels, speakers = extract_labels(all_samples)

# samples = zero_padding(samples)
samples = zero_padding(samples)
samples = normalize_data(samples)

labels = labels.astype("uint8")

#putting the parsed data into a dataframe
data = np.transpose(np.array([samples ,sample_rate,labels,speakers]))
data = pd.DataFrame(data,
		columns=['Audio_Data','Sample_Rate','Labels', 'Speakers'])
	

features = Feature_Extraction(samples)

freq , t , power = features.spect()
ZCR = features.ZCR()
print(freq[0].shape,t.shape,power.shape,ZCR.shape)
features.plot_spectogram()

print(type(freq[0]),type(t),type(power))

# print(freq[0:2])

# data["frequency"] = freq
# data.insert(2,"frequency",freq)

features_ZCR = pd.DataFrame(ZCR,
	columns= ["ZCR"])

data = pd.concat([data,features_ZCR],axis=1)	
# data.to_excel(main_dir+"/parsed_zcr.xlsx")	
# data.to_html(main_dir+"/parsed_zcr.html")

print(data.info())
plt.figure(2)
plt.hist(data["ZCR"])
plt.show()