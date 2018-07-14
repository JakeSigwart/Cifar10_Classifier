import os
import struct
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Class for loading cifar dataset and getting batches
class Cifar_dataset:
	#Load data from files. Convert to numpy array
	def __init__(self, path, training=True):
		if training:
			if os.path.isfile(path+"\\batches.meta"):
				self.classes, self.images, self.labels = load_cifar10_train(path)
			else:
				print("ERROR: Data not found in specified path.")
				self.classes = []
				self.images = []
				self.labels = []
				
		else:
			if os.path.isfile(path+"\\batches.meta"):
				self.classes, self.images, self.labels = load_cifar10_test(path)
			else:
				print("ERROR: Data not found in specified path.")
				self.classes = []
				self.images = []
				self.labels = []
		
	#Get random batch of images from the selected dataset
	def get_random_batch(self, batch_size, one_hot=True):
		self.image_batch, self.hot_label_batch = extract_random_image_batch(self.images, self.labels, [batch_size,32,32,3], one_hot, 10)
		return self.image_batch, self.hot_label_batch
	
	#Display image and label for first element in the last batch
	def display_batch_first(self):
		img = self.image_batch[0]
		lbl = self.classes(one_hot_decode(self.hot_label_batch[0]))
		plot_image(img, lbl)
	

	
def load_cifar10_train(file_path):
    raw = _unpickle_data(file_path + "\\batches.meta")[b'label_names']
    class_names = [x.decode('utf-8') for x in raw]
    print("Cifar10 class names loaded.")
    images = np.zeros(shape=[50000, 32, 32, 3], dtype=float)
    cls = np.zeros(shape=[50000], dtype=int)
    begin = 0
    for i in range(5):
        data = _unpickle_data(file_path + "\\data_batch_" + str(i + 1))
        raw_image_batch = data[b'data']
        cls_batch = np.array(data[b'labels'])
        raw_image_batch = np.array(raw_image_batch, dtype=float) / 255.0
        raw_image_batch = raw_image_batch.reshape([-1, 3, 32, 32])
        raw_image_batch = raw_image_batch.transpose([0, 2, 3, 1])
        end = begin + len(raw_image_batch)
        images[begin:end,:,:,:] = raw_image_batch
        cls[begin:end] = cls_batch
        begin = end
        print("data_batch_" + str(i+1) + " Loaded")
    return class_names, images, cls

	
def load_cifar10_test(file_path):
	raw = _unpickle_data(file_path + "\\batches.meta")[b'label_names']
	class_names = [x.decode('utf-8') for x in raw]
	print("Cifar10 class names loaded.")
	images = np.zeros(shape=[10000, 32, 32, 3], dtype=float)
	cls = np.zeros(shape=[10000], dtype=int)
	begin = 0
	data = _unpickle_data(file_path + "\\test_batch")
	raw_image_batch = data[b'data']
	cls_batch = np.array(data[b'labels'])
	raw_image_batch = np.array(raw_image_batch, dtype=float) / 255.0
	raw_image_batch = raw_image_batch.reshape([-1, 3, 32, 32])
	raw_image_batch = raw_image_batch.transpose([0, 2, 3, 1])
	end = begin + len(raw_image_batch)
	images[begin:end,:,:,:] = raw_image_batch
	cls[begin:end] = cls_batch
	print("Test data loaded.")
	return class_names, images, cls
	
#Input: An array of dimensions: [height, width, num_channels]
#Output: Display an RGB image
def plot_image(image, title=''):
	plt.axis("off")
	plt.xlabel(title)
	plt.imshow(image)
	plt.show()

#Input: A 1-D array of integer class labels, the number of classes
#Output: A 2-D array of shape: [len(class_labels), num_classes]
def one_hot_encoded(class_numbers, num_classes=None):
    if num_classes is None:
        num_classes = np.max(class_numbers) - 1
    return np.eye(num_classes, dtype=float)[class_numbers]

#Reduce one-hot-vector(s)
def one_hot_decode(val):
	if val.ndim==1:
		output = np.argmax(val)
	elif val.ndim==2:
		output = np.argmax(val, axis=1)
	else:
		print("ERROR: Input to one_hot_decode has wrong num dims")
		output = []
	return output
		
	
#Retrieving Data
#Re-constructs the object hierarchy from a file
def _unpickle_data(file_name):
    with open(file_name, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data

#Pickle up to 4 variables and store at the given file-path
def _pickle_data(file_path, var1, var2=None, var3=None, var4=None):
    output = open(file_path, 'wb')
    pickle.dump(var1, output)
    pickle.dump(var2, output)
    pickle.dump(var3, output)
    pickle.dump(var4, output)
    output.close()

#Load a numpy array from a .npy file
def load_numpy_array(file_path):
    output = np.load(file_path)
    return output

#Save a numpy array in a .npy file
def save_numpy_array(my_array, file_path):
    np.save(file_path, my_array)
    
#Load a PNG image and return as a numpy array
def load_png_as_numpy(file_path):
    img = Image.open(file_path)
    arr = np.array(img)
    return arr

#Input: data-set, integer labels, [batch_size, height, width, num_channels], number of classes (only necessary for one-hot), one-hot-output
#Output: the data batch, labels (optionally one-hot)
def extract_random_image_batch(data, labels, data_batch_shape, one_hot, num_classes=10):
    batch_size = data_batch_shape[0]
    height = data_batch_shape[1]
    width = data_batch_shape[2]
    num_channels = data_batch_shape[3]
    
    random_indexes = np.zeros(shape=[batch_size], dtype=int)
    lbls = np.zeros(shape = [batch_size], dtype=int)
    batch = np.zeros(shape= [batch_size, height, width, num_channels], dtype=float)
    for n in range(batch_size):
        random_indexes[n] = np.random.randint(0, (len(data)-1))
        batch[n,:,:,:] = data[random_indexes[n],:,:,:]
        lbls[n] = labels[random_indexes[n]]
    if one_hot:
        lbls = one_hot_encoded(lbls, num_classes)
    return batch, lbls
	