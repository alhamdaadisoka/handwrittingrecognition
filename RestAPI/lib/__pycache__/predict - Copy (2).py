#==============================================================================
#Library
#==============================================================================
import cv2
import time
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from PIL import Image
from collections import defaultdict
#==============================================================================


class HandwrittingRecognizer(object):
	is_loaded = None;
	loaded_model = None;
	file_model = 'ABC.hdf5'
	num_channel = 1
	
	def __init__(self, is_loaded=False):
		self.is_loaded = is_loaded;
		
	def from_class_to_label(self, kelas):
		list_labels = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L',
              'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h',
             'i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

		for i in list_labels:
			if kelas == list_labels.index(i):
				return i
			
			
	def potong_kalimat(self, image):
		basewidth = 1200
		im = Image.open(image)
		wpercent = (basewidth/float(im.size[0]))
		hsize = int((float(im.size[1])*float(wpercent)))
		img = im.resize((basewidth,hsize), Image.ANTIALIAS)
		img.save(image) 
	
		#load gambar
		image = cv2.imread(image)
		#ubah jadi gray
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#ubah jadi binary/threshold
		#ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
		# sharpen image
	
		# apply adaptive threshold to get black and white effect
		ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	
		
		#Menghitung total pixel
		#cek histogram
		pixels = thresh.reshape(-1,10)
	
		counts = defaultdict(int)
		for pixel in pixels:
			if pixel[0] == pixel[1] == pixel[2]:
				counts[pixel[0]] += 1
	
		total_1 = 0
		total_2 = 0
	
		for index, pv in enumerate(sorted(counts.keys())):
			print("(%d,%d,%d): %d pixels" % (pv, pv, pv, counts[pv]))
			
			if index==0:
				total_1 = counts[pv]
			else:
				total_2 = counts[pv]
				
		if total_2 > total_1:
			ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
			
		#lakukan dilasi
		kernel = np.ones((15,100), np.uint8)
		dilasi = cv2.dilate(thresh, kernel, iterations=1)
		
		hasil_deteksi = ''
			
		#cari contour
		im2, ctrs, hier = cv2.findContours(dilasi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		#sort contour
		sorted_ctrs = sorted(ctrs, key=lambda ctr:  cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1])
		
		for i, ctr in enumerate(sorted_ctrs):
			
			#ambil bounding box
			x, y, w, h = cv2.boundingRect(ctr)
			
			#ambil ROI
			roi = thresh[y:y+h, x:x+w]
			
			if w > 5 and h > 15:
				#cv2.imwrite('output\\{}.png'.format(i), roi)
				hasil_deteksi = hasil_deteksi + self.potong_kata(roi) + '<br/>'
		  
		return hasil_deteksi

	def potong_kata(self, image):

		hasil_deteksi = ''
		
		kernel = np.ones((15,20), np.uint8)
		dilasi = cv2.dilate(image, kernel, iterations=1)
		
		#cari contour
		im2, ctrs, hier = cv2.findContours(dilasi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		#sort contour
		sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1])
		
		for i, ctr in enumerate(sorted_ctrs):
			
			#ambil bounding box
			x, y, w, h = cv2.boundingRect(ctr)
			
			#ambil ROI
			roi1 = image[y:y+h, x:x+w]
			
			if w > 5 and h > 15:
				hasil_deteksi = hasil_deteksi + self.potong_huruf(roi1) + ' '
				
		return hasil_deteksi


	def potong_huruf(self, image):
		
		hasil_deteksi = ''
		
		kernel = np.ones((10,2), np.uint8)
		dilasi = cv2.dilate(image, kernel, iterations=1)
		
		#cari contour
		im2, ctrs, hier = cv2.findContours(dilasi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		#sort contour
		sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1])
		
		for i, ctr in enumerate(sorted_ctrs):
			
			#ambil bounding box
			x, y, w, h = cv2.boundingRect(ctr)
			
			#ambil ROI
			roi = image[y:y+h, x:x+w]
			
			if w > 5 and h > 15:
				hasil_deteksi = hasil_deteksi + self.get_prediksi(roi)
				
		return hasil_deteksi
	

	def get_prediksi(self, image):
		hasil_deteksi = ''
		
		test_image=cv2.resize(image,(28,28))
		test_image = np.array(test_image)
		test_image = test_image.astype('float64')
		test_image /= 255
	
		if self.num_channel==1:
			if K.image_dim_ordering()=='th':
				test_image= np.expand_dims(test_image, axis=0)
				test_image= np.expand_dims(test_image, axis=0)
			else:
				test_image= np.expand_dims(test_image, axis=3) 
				test_image= np.expand_dims(test_image, axis=0)
		else:
			if K.image_dim_ordering()=='th':
				test_image=np.rollaxis(test_image,2,0)
				test_image= np.expand_dims(test_image, axis=0)
			else:
				test_image= np.expand_dims(test_image, axis=0)
	
		deteksi = self.loaded_model.predict_classes(test_image)[0]
		deteksi = self.from_class_to_label(deteksi)
	
		hasil_deteksi = hasil_deteksi + str(deteksi)
		return hasil_deteksi


	def predict(self, image):
		start_time = time.time()
		
		if self.is_loaded==False:
			self.loaded_model = load_model(self.file_model)
			self.is_loaded = True
		
		#lakukan prediksi
		hasil_deteksi = self.potong_kalimat(image)
		
		elapsed_time = time.time() - start_time
		
		print("Elapsed time: {}".format(elapsed_time))
		return hasil_deteksi
