import os
import pickle
import tensorflow as tf
import numpy as np
import argparse
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG19, VGG16
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from imgaug import augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch
import imgaug as ia
from tqdm import tqdm as tqdm_base
import csv


def tqdm(*args, **kwargs):
	if hasattr(tqdm_base, '_instances'):
		for instance in list(tqdm_base._instances):
			tqdm_base._decr_instances(instance)
	return tqdm_base(*args, **kwargs)

def batch_data_generation(name, names, data_name_ind):
		
		
		#download the whole track for a given cyclone's name
		pi_name = "./data/Cuts/" + name
		with open(pi_name, 'rb') as fp:
			cyclone_track = pickle.load(fp)
		c_min = 92352
		c_max = 105111	
		#normalisation constants
		#normalize each array between 0 and 1
		cyclone_track_data = [cyclone_track[i][0][np.newaxis,:,:] for i in range(len(cyclone_track))]
		cyclone_track_data = np.concatenate(cyclone_track_data, axis=0)
		cyclone_track_data_scaled = (cyclone_track_data-c_min)/(c_max-c_min)        
		
		#lists for all batches 
		X_1 = []
		X_2 = []
		y = []
		
		
		for i in range(len(cyclone_track_data_scaled)-1):
			X_1.append(cyclone_track_data_scaled[i])
			y.append(1)
			
		# fill out the first half of the batch with data that we get at step t + 1 (label 1)
		for i in range(0, len(cyclone_track_data_scaled)-1, 1):
			X_2.append(cyclone_track_data_scaled[i+1])

		#fill out the second half of the batch, (label 0)
		for i in range(0,len(cyclone_track_data_scaled)-1,1):
		
			ex = data_name_ind.loc[data_name_ind['datetime'] == cyclone_track[i+1][1]]
			for m in ex['name']:
				if m != name:
					local_name = "./data/Cuts/" + m
					with open(local_name, 'rb') as ln:
						local_track = pickle.load(ln)

						#normilize loca_track
						cyclone_local_data = [local_track[i][0][np.newaxis,:,:] for i in range(len(local_track))]
						cyclone_local_data = np.concatenate(cyclone_local_data, axis=0)
						cyclone_local_data_scaled = (cyclone_local_data-c_min)/(c_max-c_min) 

						X_1.append(cyclone_track_data_scaled[i])
						X_2.append(cyclone_local_data_scaled[int(ex[ex['name']==m]['index'])])
						y.append(0)
				

		return np.asarray(X_1), np.asarray(X_2), np.asarray(y)
	
class TCNN(object):
	def __init__(self, input_shape = (80,80,1), learning_rate = 0.00001, epochs = 13):
		self.input_shape = input_shape
		self.lr = learning_rate 
		self.epochs = epochs
	
	def get_siamese_model(self):
		left_input = Input(self.input_shape)
		right_input = Input(self.input_shape)

		encoder = Sequential()
		encoder.add(Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape, padding='same',
										kernel_initializer="he_uniform", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape, padding='same',
										kernel_initializer="he_uniform", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(32, (3,3), strides=(2,2), activation='relu', input_shape=self.input_shape, padding='same',
										kernel_initializer="he_uniform", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(64, (3,3), activation='relu',  padding='same',
											kernel_initializer="he_uniform",
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(64, (3,3), activation='relu',  padding='same',
											kernel_initializer="he_uniform",
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(64, (3,3), strides=(2,2), activation='relu',  padding='same',
											kernel_initializer="he_uniform",
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(128, (3,3), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(128, (3,3), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(128, (3,3), strides=(2,2), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		
		encoder.add(Conv2D(256, (3,3), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(256, (3,3), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(256, (3,3), strides=(2,2), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		
		encoder.add(Flatten())
		encoder.add(Dense(2048, activation='relu',
									 kernel_regularizer=l2(1e-3),
										kernel_initializer="he_uniform",    bias_initializer="Ones"))
		encoder.add(Dense(1024, activation='relu',
									 kernel_regularizer=l2(1e-3),
										kernel_initializer="he_uniform",    bias_initializer="Ones"))
		encoder.add(Dense(512, activation='relu',
									 kernel_regularizer=l2(1e-3),
										kernel_initializer="he_uniform",    bias_initializer="Ones"))
		
		encoded_l = encoder(left_input)
		encoded_r = encoder(right_input)
		
		L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
		L1_distance = L1_layer([encoded_l, encoded_r])
		
		d1 = Dense(128, activation='relu', kernel_regularizer=l2(1e-3), kernel_initializer="he_uniform", bias_initializer="Ones")(L1_distance)
		d2 = Dense(16, activation='relu', kernel_regularizer=l2(1e-3), kernel_initializer="he_uniform", bias_initializer="Ones")(d1)
		
		prediction = Dense(1, activation='sigmoid', bias_initializer="Ones")(d2)
		
		sim_model = Model(inputs=[left_input,right_input],outputs=prediction)

		return sim_model
		
	def train(self, df_file):
		with open(df_file, 'rb') as fp:
			df = pickle.load(fp)
		test_names = np.unique(df[df['datetime'] >= '2010-01-01']['name'])
		train_names = np.unique(df[df['datetime'] < '2010-01-01']['name'])
		
		sim_model = self.get_siamese_model()		
		optimizer = Adam(lr = self.lr)
		sim_model.compile(loss="binary_crossentropy", optimizer=optimizer)
		
		self.losses = []
		self.validation_losses = []
		
		for epoch in range(self.epochs):
			
			print('learning %d epoch' % epoch)
			local_train = []
			
			for name in tqdm(train_names[:]):
				try:
					X_1, X_2, y = batch_data_generation(name, train_names[:],df)
					X_1 = np.asarray([np.expand_dims(x,-1) for x in X_1])
					X_2 = np.asarray([np.expand_dims(x,-1) for x in X_2])
					loss = sim_model.train_on_batch([X_1, X_2], y)
					local_train.append(loss)	
				except:
					pass
			
			print("Train loss on %d epoch: %.2f" % (epoch, np.mean(local_train)))
			self.losses.append(np.mean(local_train))
				
			local_val = []
			for name in tqdm(test_names[:]):
				try:
					X_1_pred, X_2_pred, y = batch_data_generation(name, test_names[:],df)
					X_1_pred = np.asarray([np.expand_dims(x,-1) for x in X_1_pred])
					X_2_pred = np.asarray([np.expand_dims(x,-1) for x in X_2_pred])
					loss = sim_model.test_on_batch([X_1_pred, X_2_pred],y)
					local_val.append(loss)
				except:
					pass
			print("Validation loss on %d epoch: %.2f: " % (epoch, np.mean(local_val)))	
			self.validation_losses.append(np.mean(local_val))
		
		save_tr = pd.DataFrame(self.losses)
		csv_data = save_tr.to_csv(index=False)
		save_tr.to_csv('./logs/tcnn/train_loss.csv', index=False)
		
		save_vl = pd.DataFrame(self.validation_losses)
		csv_data = save_vl.to_csv(index=False)
		save_vl.to_csv('./logs/tcnn/validation_loss.csv', index=False)

#------------------------------------------------------------------------------------------------------------------------------------------------


class MCNN(object):
	def __init__(self, input_shape = (200,200,2), learning_rate = 0.00001, epochs = 300):
		self.input_shape = input_shape
		self.lr = learning_rate 
		self.epochs = epochs
		
	def get_siamese_model(self):

		left_input = Input(self.input_shape)
		right_input = Input(self.input_shape)

		encoder = Sequential()
		encoder.add(Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape, padding='same',
										kernel_initializer="he_uniform", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape, padding='same',
										kernel_initializer="he_uniform", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(32, (3,3), strides=(2,2), activation='relu', input_shape=self.input_shape, padding='same',
										kernel_initializer="he_uniform", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(64, (3,3), activation='relu',  padding='same',
											kernel_initializer="he_uniform",
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(64, (3,3), activation='relu',  padding='same',
											kernel_initializer="he_uniform",
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(64, (3,3), strides=(2,2), activation='relu',  padding='same',
											kernel_initializer="he_uniform",
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(128, (3,3), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(128, (3,3), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(128, (3,3), strides=(2,2), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		
		encoder.add(Conv2D(256, (3,3), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(256, (3,3), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(256, (3,3), strides=(2,2), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Flatten())
		encoder.add(Dense(1024, activation='relu',
									 kernel_regularizer=l2(1e-3),
										kernel_initializer="he_uniform",    bias_initializer="Ones"))
		encoder.add(Dense(512, activation='relu',
									 kernel_regularizer=l2(1e-3),
										kernel_initializer="he_uniform",    bias_initializer="Ones"))
		encoder.add(Dense(224, activation='relu',
									 kernel_regularizer=l2(1e-3),
										kernel_initializer="he_uniform",    bias_initializer="Ones"))
		
		encoded_l = encoder(left_input)
		encoded_r = encoder(right_input)
		
		L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
		L1_distance = L1_layer([encoded_l, encoded_r])
		
		d1 = Dense(128, activation='relu', kernel_regularizer=l2(1e-3), kernel_initializer="he_uniform", bias_initializer="Ones")(L1_distance)
		d2 = Dense(16, activation='relu', kernel_regularizer=l2(1e-3), kernel_initializer="he_uniform", bias_initializer="Ones")(d1)
		
		prediction = Dense(1, activation='sigmoid', bias_initializer="Ones")(d2)
		
		sim_model = Model(inputs=[left_input,right_input],outputs=prediction)
		
		return sim_model

			
	def train(self):
		
		sim_model = self.get_siamese_model()
		#K.clear_session()
		#tf.reset_default_graph()
		#os.environ["CUDA_VISIBLE_DEVICES"]="0"
		#config = tf.ConfigProto()
		#config.gpu_options.allow_growth = True
		#tf.keras.backend.set_session(tf.Session(config=config))
		
		optimizer = Adam(lr = self.lr)
		sim_model.compile(loss="binary_crossentropy", optimizer=optimizer)
		
		self.losses = []
		self.validation_losses = []

		for epoch in range(self.epochs):
			
			print('learning %d epoch' % epoch)
			local_train = []
			
			for name in tqdm(list(range(1,1201))): #train_names 
			
				j = np.random.randint(0,20)
				pi_name = "./data/datasets/train/" + str(name) + "_" + str(j)
				with open(pi_name, 'rb') as fp:
					aug_data = pickle.load(fp)
				
				try:	
					y = aug_data[4]
					X_1_WV_aug_prep = preprocess_input(aug_data[0])[:,:,:,0]
					X_2_WV_aug_prep = preprocess_input(aug_data[1])[:,:,:,0]
					X_1_IR_aug_prep = preprocess_input(aug_data[2])[:,:,:,0]
					X_2_IR_aug_prep = preprocess_input(aug_data[3])[:,:,:,0]

					X_1_WV_aug_prep = np.asarray([np.expand_dims(x,-1) for x in X_1_WV_aug_prep])
					X_2_WV_aug_prep = np.asarray([np.expand_dims(x,-1) for x in X_2_WV_aug_prep])
					X_1_IR_aug_prep = np.asarray([np.expand_dims(x,-1) for x in X_1_IR_aug_prep])
					X_2_IR_aug_prep = np.asarray([np.expand_dims(x,-1) for x in X_2_IR_aug_prep])


					concat_X1_aug = np.concatenate([X_1_IR_aug_prep, X_1_WV_aug_prep], axis=3)
					concat_X2_aug = np.concatenate([X_2_IR_aug_prep, X_2_WV_aug_prep], axis=3)

					loss = sim_model.train_on_batch([concat_X1_aug, concat_X2_aug], y)
					local_train.append(loss)

				except:
					pass

			print("Train loss on %d epoch: %.2f" % (epoch, np.mean(local_train)))
			self.losses.append(np.mean(local_train))
				
			local_val = []
			
			for name_val in tqdm(range(1201,1401)):
				
				try:
					
					pi_name = "./data/datasets/validation/" + str(name_val) 
					
					with open(pi_name, 'rb') as fp:
						aug_data = pickle.load(fp)
							
					y = aug_data[4]
					X_1_WV_prep = preprocess_input(aug_data[0])[:,:,:,0]
					X_2_WV_prep = preprocess_input(aug_data[1])[:,:,:,0]
					X_1_IR_prep = preprocess_input(aug_data[2])[:,:,:,0]
					X_2_IR_prep = preprocess_input(aug_data[3])[:,:,:,0]

					X_1_WV = np.asarray([np.expand_dims(x,-1) for x in X_1_WV_prep])
					X_2_WV = np.asarray([np.expand_dims(x,-1) for x in X_2_WV_prep])
					X_1_IR = np.asarray([np.expand_dims(x,-1) for x in X_1_IR_prep])
					X_2_IR = np.asarray([np.expand_dims(x,-1) for x in X_2_IR_prep])


					concat_X1 = np.concatenate([X_1_IR, X_1_WV], axis=3)
					concat_X2 = np.concatenate([X_2_IR, X_2_WV], axis=3)


					val_loss = sim_model.test_on_batch([concat_X1, concat_X2], y)
					local_val.append(val_loss) 
						
				except:
					pass
				
			print("Validation loss on %d epoch: %.2f: " % (epoch, np.mean(local_val)))
			self.validation_losses.append(np.mean(local_val))
			
		save_tr = pd.DataFrame(self.losses)
		csv_data = save_tr.to_csv(index=False)
		save_tr.to_csv('./logs/mcnn/train_loss.csv', index=False)

		save_vl = pd.DataFrame(self.validation_losses)
		csv_data = save_vl.to_csv(index=False)
		save_vl.to_csv('./logs/mcnn/validation_loss.csv', index=False)
			

#-------------------------------------------------------------------------------------------------------------------


class MCNNd(object):
	def __init__(self, input_shape = (200,200,2), learning_rate = 0.00001, epochs = 300):
		self.input_shape = input_shape
		self.lr = learning_rate 
		self.epochs = epochs
	
	def get_siamese_model(self):

		left_input = Input(self.input_shape)
		right_input = Input(self.input_shape)
		distance = Input(shape=(1, ))

		encoder = Sequential()
		encoder.add(Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape, padding='same',
										kernel_initializer="he_uniform", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape, padding='same',
										kernel_initializer="he_uniform", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(32, (3,3), strides=(2,2), activation='relu', input_shape=self.input_shape, padding='same',
										kernel_initializer="he_uniform", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(64, (3,3), activation='relu',  padding='same',
											kernel_initializer="he_uniform",
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(64, (3,3), activation='relu',  padding='same',
											kernel_initializer="he_uniform",
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(64, (3,3), strides=(2,2), activation='relu',  padding='same',
											kernel_initializer="he_uniform",
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(128, (3,3), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(128, (3,3), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(128, (3,3), strides=(2,2), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		
		encoder.add(Conv2D(256, (3,3), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(256, (3,3), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Conv2D(256, (3,3), strides=(2,2), activation='relu',  kernel_initializer="he_uniform",  padding='same',
												 bias_initializer="Ones", kernel_regularizer=l2(2e-4)))
		encoder.add(Flatten())
		encoder.add(Dense(1024, activation='relu',
									 kernel_regularizer=l2(1e-3),
										kernel_initializer="he_uniform",    bias_initializer="Ones"))
		encoder.add(Dense(512, activation='relu',
									 kernel_regularizer=l2(1e-3),
										kernel_initializer="he_uniform",    bias_initializer="Ones"))
		encoder.add(Dense(224, activation='relu',
									 kernel_regularizer=l2(1e-3),
										kernel_initializer="he_uniform",    bias_initializer="Ones"))
		
		encoded_l = encoder(left_input)
		encoded_r = encoder(right_input)
		
		L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
		L1_distance = L1_layer([encoded_l, encoded_r])
		
		feature = Sequential()
		feature.add(Dense(30, activation='relu',
									 kernel_regularizer=l2(1e-1),
										kernel_initializer="he_uniform",    bias_initializer="Ones"))
		feature_dist = feature(distance)
		
		L1_distdanse_dist = concatenate([L1_distance, feature_dist])
		
		d1 = Dense(254, activation='relu', kernel_regularizer=l2(1e-1), kernel_initializer="he_uniform", bias_initializer="Ones")(L1_distdanse_dist)
		d2 =Dense(128, activation='relu', kernel_regularizer=l2(1e-1), kernel_initializer="he_uniform", bias_initializer="Ones")(d1)
		d3 = Dense(16, activation='relu', kernel_regularizer=l2(1e-1), kernel_initializer="he_uniform", bias_initializer="Ones")(d2)
		
		prediction = Dense(1, activation='sigmoid', bias_initializer="Ones")(d3)
		
		sim_model = Model(inputs=[left_input,right_input, distance],outputs=prediction)
		
		return sim_model

			
	def train(self):
		
		sim_model = self.get_siamese_model()
		#K.clear_session()
		#tf.reset_default_graph()
		#os.environ["CUDA_VISIBLE_DEVICES"]="0"
		#config = tf.ConfigProto()
		#config.gpu_options.allow_growth = True
		#tf.keras.backend.set_session(tf.Session(config=config))
		
		optimizer = Adam(lr = self.lr)
		sim_model.compile(loss="binary_crossentropy", optimizer=optimizer)
		
		self.losses = []
		self.validation_losses = []

		for epoch in range(self.epochs):
			
			local_train = []
			
			for name in tqdm(list(range(1,1201))):
				
				
				j = np.random.randint(0,20)
				pi_name = "./data/datasets/train_dist/" + str(name) + "_" + str(j)
				with open(pi_name, 'rb') as fp:
					aug_data = pickle.load(fp)
				
				try:

					y = aug_data[4]
					dist_input = aug_data[5]


					X_1_WV_aug_prep = preprocess_input(aug_data[0])[:,:,:,0]
					X_2_WV_aug_prep = preprocess_input(aug_data[1])[:,:,:,0]
					X_1_IR_aug_prep = preprocess_input(aug_data[2])[:,:,:,0]
					X_2_IR_aug_prep = preprocess_input(aug_data[3])[:,:,:,0]

					X_1_WV_aug_prep = np.asarray([np.expand_dims(x,-1) for x in X_1_WV_aug_prep])
					X_2_WV_aug_prep = np.asarray([np.expand_dims(x,-1) for x in X_2_WV_aug_prep])
					X_1_IR_aug_prep = np.asarray([np.expand_dims(x,-1) for x in X_1_IR_aug_prep])
					X_2_IR_aug_prep = np.asarray([np.expand_dims(x,-1) for x in X_2_IR_aug_prep])


					concat_X1_aug = np.concatenate([X_1_IR_aug_prep, X_1_WV_aug_prep], axis=3)
					concat_X2_aug = np.concatenate([X_2_IR_aug_prep, X_2_WV_aug_prep], axis=3)

					loss = sim_model.train_on_batch([concat_X1_aug, concat_X2_aug, dist_input], y)

					local_train.append(loss)

				except:
					pass

			print("Train loss on %d epoch: %.2f" % (epoch, np.mean(local_train)))
			
			self.losses.append(np.mean(local_train))
				
				

			local_val = []
			
			for name_val in tqdm(range(1201,1401)):
				
				try:
					
					pi_name = "./data/datasets/validation_dist/" + str(name_val) 
					
					with open(pi_name, 'rb') as fp:
						aug_data = pickle.load(fp)
							
					y = aug_data[4]
					dist_input = aug_data[5]

					X_1_WV_prep = preprocess_input(aug_data[0])[:,:,:,0]
					X_2_WV_prep = preprocess_input(aug_data[1])[:,:,:,0]
					X_1_IR_prep = preprocess_input(aug_data[2])[:,:,:,0]
					X_2_IR_prep = preprocess_input(aug_data[3])[:,:,:,0]

					X_1_WV = np.asarray([np.expand_dims(x,-1) for x in X_1_WV_prep])
					X_2_WV = np.asarray([np.expand_dims(x,-1) for x in X_2_WV_prep])
					X_1_IR = np.asarray([np.expand_dims(x,-1) for x in X_1_IR_prep])
					X_2_IR = np.asarray([np.expand_dims(x,-1) for x in X_2_IR_prep])


					concat_X1 = np.concatenate([X_1_IR, X_1_WV], axis=3)
					concat_X2 = np.concatenate([X_2_IR, X_2_WV], axis=3)


					val_loss = sim_model.test_on_batch([concat_X1, concat_X2, dist_input], y)
					local_val.append(val_loss) 
						
				except:
					pass
				
			print("Validation loss on %d epoch: %.2f: " % (epoch, np.mean(local_val)))
				
			self.validation_losses.append(np.mean(local_val))
			
		save_tr = pd.DataFrame(self.losses)
		csv_data = save_tr.to_csv(index=False)
		save_tr.to_csv('./logs/mcnnd/train_loss.csv', index=False)

		save_vl = pd.DataFrame(self.validation_losses)
		csv_data = save_vl.to_csv(index=False)
		save_vl.to_csv('./logs/mcnnd/validation_loss.csv', index=False)
				
		



