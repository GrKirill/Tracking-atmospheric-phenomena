import pickle
from netCDF4 import Dataset
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import pandas as pd
import os
from datetime import datetime
from libs.service import *
from libs.quality_metric import calculate_distance
from tqdm import tqdm as tqdm_base


def tqdm(*args, **kwargs):
	if hasattr(tqdm_base, '_instances'):
		for instance in list(tqdm_base._instances):
			tqdm_base._decr_instances(instance)
	return tqdm_base(*args, **kwargs)

def min_max_IR():
	c_min_IR = 10000
	c_max_IR = 0
	for name in tqdm(train_names[:]):
		pi_name = "./data/Cuts_Mesocyclones/" + str(name)
		with open(pi_name, 'rb') as fp:
			cyclone_track = pickle.load(fp)
			for i in range(len(cyclone_track[0])):
				for j in cyclone_track[0][i].ravel():
					if j < c_min_IR and j != 0.0:
						c_min_IR = j
			for i in range(len(cyclone_track[0])):
				for j in cyclone_track[0][i].ravel():
					if j > c_max_IR:
						c_max_IR = j
	return min_IR, max_IR

def min_max_WN():	
	c_min_WV = 10000
	c_max_WV = 0
	for name in tqdm(train_names[:]):
		pi_name = "./data/Cuts_Mesocyclones/" + str(name)
		with open(pi_name, 'rb') as fp:
			cyclone_track = pickle.load(fp)
			for i in range(len(cyclone_track[1])):
				for j in cyclone_track[1][i].ravel():
					if j < c_min_WV and j != 0.0:
						c_min_WV = j
			for i in range(len(cyclone_track[1])):
				for j in cyclone_track[1][i].ravel():
					if j > c_max_WV:
						c_max_WV = j
	return min_WV, max_WV
	
def pad_with(vector, pad_width, iaxis, kwargs):
	pad_value = kwargs.get('padder', 0)
	vector[:pad_width[0]] = pad_value
	vector[-pad_width[1]:] = pad_value
	

def df(database_file, names):
	database = pd.read_csv(database_file, sep = ";", header = None)
	df = pd.DataFrame(columns=['datetime', 'name', 'index'])
	for name in tqdm(names[:]):
		for i in range(len(database[database[2] == name])):
			date = str(int(database[database[2] == name].iloc[i][3]))
			long = database[database[2] == name].iloc[i][4]
			lat = database[database[2] == name].iloc[i][5]
			df = df.append({'datetime' : datetime.strptime(date, '%Y%m%d%H'),
							'name':name ,'index' : i, 'long' : long, 'lat' : lat } , ignore_index=True)
	df = df.sort_values(by='datetime')
	with open('./data/df_mesocyclones.p', 'wb') as fp:
		pickle.dump(df , fp, protocol=pickle.HIGHEST_PROTOCOL)
			

def create_cuts(database_file,  path_to_mosaics):
	
	database = pd.read_csv(database_file, sep = ";", header = None)
	last_track_number = np.unique(database.iloc[:,2])[-1]

	for i in tqdm(range(1,last_track_number + 1)):
		track = database[database.iloc[:,2] == i]
		times = [str(i) for i in track[3]]
		longs = list(track[4])
		lats = list(track[5])
		
		list_for_images_IR = []
		list_for_images_WV = []
		for j in range(len(times)):
			#ex.: times[i] = '2004090221' YYYY MM DD HH
			#/storage/tartar/DATA/SATELLITE_OBS/MOSAICS/Antarctic
			to_open_IR = path_to_mosaics + "/IR/" + times[j][:4] + "/" + "Antarctic.Composite.5km.Infrared." + times[j][:4] + "." + times[j][4:6] + "." + times[j][6:8] + "." + times[j][8:11] + "Z.nc"
			to_open_WV = path_to_mosaics + "/WV/" + times[j][:4] + "/" + "Antarctic.Composite.5km.WaterVapor." + times[j][:4] + "." + times[j][4:6] + "." + times[j][6:8] + "." + times[j][8:11] + "Z.nc"		
			try:
				ds_IR = Dataset(to_open_IR, 'r')
				ds_WV = Dataset(to_open_WV, 'r')

				IR_data = ds_IR.variables['data'][:]
				WV_data = ds_WV.variables['data'][:]

				rho_IR = np.sqrt((ds_IR.variables['lat'][:] - (lats[j]))**2 + (ds_IR.variables['lon'][:]-(longs[j]))**2)
				center_pt_IR = np.unravel_index(np.argmin(rho_IR), rho_IR.shape)
				cut_IR = np.pad(np.squeeze(IR_data), 120, pad_with)[120 + (center_pt_IR[0] - 100):center_pt_IR[0]+ 100 + 120, 120+(center_pt_IR[1]-100):center_pt_IR[1]+100+120]
					
				rho_WV = np.sqrt((ds_WV.variables['lat'][:] - (lats[j]))**2 + (ds_WV.variables['lon'][:]-(longs[j]))**2)
				center_pt_WV = np.unravel_index(np.argmin(rho_WV), rho_WV.shape)
					
				cut_WV = np.pad(np.squeeze(WV_data), 120, pad_with)[120 + (center_pt_WV[0] - 100):center_pt_WV[0]+100 + 120, 120+(center_pt_WV[1]-100):center_pt_WV[1]+100+120]
						
				list_for_images_IR.append(cut_IR)
				list_for_images_WV.append(cut_WV)

				ds_IR.close()
				ds_WV.close()
			except:
				pass
		
		joined = [list_for_images_IR, list_for_images_WV]
		pi_name = "./data/Cuts_Mesocyclones/" + str(i) 
					
		with open(pi_name, 'wb') as fp:
			pickle.dump(joined, fp, protocol=pickle.HIGHEST_PROTOCOL)
		
		
def MCNND_batch_data_generation_IR(name, c_min_IR, c_max_IR, database_file, df_file):
	
	database = pd.read_csv(database_file, sep = ";", header = None)
	with open(df_file, 'rb') as fp:
		df = pickle.load(fp)
	name_real_lat_lons = [(database[database.iloc[:,2] == name][4].values[i], database[database.iloc[:,2] == name][5].values[i]) 
						  for i in range(len(database[database.iloc[:,2] == name]))]
	
	distances = []
	#download the whole track for a given cyclone's name
	pi_name = "./data/Cuts_Mesocyclones/" + str(name)
	with open(pi_name, 'rb') as fp:
		cyclone_track = pickle.load(fp)
		
	#normalize each array between 0 and 1
	cyclone_track_data = [cyclone_track[0][i][np.newaxis,:,:] for i in range(len(cyclone_track[0]))]
	cyclone_track_data = np.concatenate(cyclone_track_data, axis=0)
	cyclone_track_data_scaled = (cyclone_track_data-c_min_IR)/(c_max_IR-c_min_IR) 
	#filtering
	cyclone_track_data_scaled.ravel()[cyclone_track_data_scaled.ravel() < 0] = 0
	
	#lists for all batches 
	X_1 = []
	X_2 = []
	y = []


	for i in range(len(cyclone_track_data_scaled)-1):
		X_1.append(cyclone_track_data_scaled[i])
		y.append(1)
		distances.append(calculate_distance(name_real_lat_lons[i+1][1],name_real_lat_lons[i+1][0],name_real_lat_lons[i][1], name_real_lat_lons[i][0]))

	#fill out the first half of the batch with data that we get at step t + 1 (label 1)
	for i in range(0, len(cyclone_track_data_scaled)-1, 1):
		X_2.append(cyclone_track_data_scaled[i+1])

	#fill out the second half of the batch, (label 0)
	for i in range(0,int(len(cyclone_track_data_scaled))-1,1):
		
		database_srez = database[database[2] == name]
		time_next = datetime.strptime(str(int(database_srez[3].iloc[i+1])), '%Y%m%d%H')
		ex = df.loc[df['datetime'] == time_next]
		for m in ex['name']:
			if m != name:
				local_name = "./data/Cuts_Mesocyclones/" + str(m)
				with open(local_name, 'rb') as ln:
					local_track = pickle.load(ln)
					#normalize loca_track
					cyclone_local_data = [local_track[0][k][np.newaxis,:,:] for k in range(len(local_track[0]))]
					cyclone_local_data = np.concatenate(cyclone_local_data, axis=0)
					cyclone_local_data_scaled_BLYAT = (cyclone_local_data-c_min_IR)/(c_max_IR-c_min_IR) 
					#filtering
					cyclone_local_data_scaled_BLYAT.ravel()[cyclone_local_data_scaled_BLYAT.ravel() < 0] = 0
					try:
						
						X_1.append(cyclone_track_data_scaled[i])
						X_2.append(cyclone_local_data_scaled_BLYAT[int(ex[ex['name']== m]['index'])])
						y.append(0)
						fake_long = database[database.iloc[:,2] == m][4].values[int(ex[ex['name']== m]['index'])]
						fake_lat = database[database.iloc[:,2] == m][5].values[int(ex[ex['name']== m]['index'])]
						distances.append(calculate_distance(name_real_lat_lons[i][1], name_real_lat_lons[i][0],fake_lat, fake_long))           
						break
					except:
						pass

	return np.asarray(X_1), np.asarray(X_2), np.asarray(y), np.array(distances)


def MCNND_batch_data_generation_WV(name, c_min_WV, c_max_WV, database_file):
	
	database = pd.read_csv(database_file, sep = ";", header = None)
	with open('./data/df_mesocyclones.p', 'rb') as fp:
		df = pickle.load(fp)
	#download the whole track for a given cyclone's name
	pi_name = "./data/Cuts_Mesocyclones/" + str(name)
	with open(pi_name, 'rb') as fp:
		cyclone_track = pickle.load(fp)

	#normalize each array between 0 and 1
	cyclone_track_data = [cyclone_track[1][i][np.newaxis,:,:] for i in range(len(cyclone_track[1]))]
	cyclone_track_data = np.concatenate(cyclone_track_data, axis=0)
	cyclone_track_data_scaled = (cyclone_track_data-c_min_WV)/(c_max_WV-c_min_WV) 
	#filtering
	cyclone_track_data_scaled.ravel()[cyclone_track_data_scaled.ravel() < 0] = 0

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
	for i in range(0,int(len(cyclone_track_data_scaled))-1,1):
		
		database_srez = database[database[2] == name]
		time_next = datetime.strptime(str(int(database_srez[3].iloc[i+1])), '%Y%m%d%H')
		ex = df.loc[df['datetime'] == time_next]
		for m in ex['name']:
			if m != name:
				local_name = "./data/Cuts_Mesocyclones/" + str(m)
				with open(local_name, 'rb') as ln:
					local_track = pickle.load(ln)

					#normilize local_track
					cyclone_local_data = [local_track[1][k][np.newaxis,:,:] for k in range(len(local_track[1]))]
					cyclone_local_data = np.concatenate(cyclone_local_data, axis=0)
					cyclone_local_data_scaled_BLYAT = (cyclone_local_data-c_min_WV)/(c_max_WV-c_min_WV) 
					#filtering
					cyclone_local_data_scaled_BLYAT.ravel()[cyclone_local_data_scaled_BLYAT.ravel() < 0] = 0
					try:
						X_1.append(cyclone_track_data_scaled[i])
						X_2.append(cyclone_local_data_scaled_BLYAT[int(ex[ex['name']== m]['index'])])
						y.append(0)	
						break
					except:
						pass

	return np.asarray(X_1), np.asarray(X_2), np.asarray(y)
	
def MCNN_batch_data_generation_IR(name, c_min_IR, c_max_IR, database_file):
	
	database = pd.read_csv(database_file, sep = ";", header = None)
	with open('./data/df_mesocyclones.p', 'rb') as fp:
		df = pickle.load(fp)
	pi_name = "./data/Cuts_Mesocyclones/" + str(name)
	with open(pi_name, 'rb') as fp:
		cyclone_track = pickle.load(fp)

	#normalize each array between 0 and 1
	cyclone_track_data = [cyclone_track[0][i][np.newaxis,:,:] for i in range(len(cyclone_track[0]))]
	cyclone_track_data = np.concatenate(cyclone_track_data, axis=0)
	cyclone_track_data_scaled = (cyclone_track_data-c_min_IR)/(c_max_IR-c_min_IR) 
	#filtering
	cyclone_track_data_scaled.ravel()[cyclone_track_data_scaled.ravel() < 0] = 0

	#lists for all batches 
	X_1 = []
	X_2 = []
	y = []

	for i in range(len(cyclone_track_data_scaled)-1):
		X_1.append(cyclone_track_data_scaled[i])
		y.append(1)

	#fill out the first half of the batch with data that we get at step t + 1 (label 1)
	for i in range(0, len(cyclone_track_data_scaled)-1, 1):
		X_2.append(cyclone_track_data_scaled[i+1])

	#fill out the second half of the batch, (label 0)
	for i in range(0,int(len(cyclone_track_data_scaled))-1,1):
		
		database_srez = database[database[2] == name]
		time_next = datetime.strptime(str(int(database_srez[3].iloc[i+1])), '%Y%m%d%H')
		ex = df.loc[df['datetime'] == time_next]
		for m in ex['name']:
			if m != name:
				local_name = "./data/Cuts_Mesocyclones/" + str(m)
				with open(local_name, 'rb') as ln:
					local_track = pickle.load(ln)
					
					#normilize local_track
					cyclone_local_data = [local_track[0][k][np.newaxis,:,:] for k in range(len(local_track[0]))]
					cyclone_local_data = np.concatenate(cyclone_local_data, axis=0)
					cyclone_local_data_scaled_BLYAT = (cyclone_local_data-c_min_IR)/(c_max_IR-c_min_IR) 
					cyclone_local_data_scaled_BLYAT.ravel()[cyclone_local_data_scaled_BLYAT.ravel() < 0] = 0
					try:
						X_1.append(cyclone_track_data_scaled[i])
						X_2.append(cyclone_local_data_scaled_BLYAT[int(ex[ex['name']== m]['index'])])
						y.append(0)
						break
					except:
						pass

	return np.asarray(X_1), np.asarray(X_2), np.asarray(y)
	
def MCNN_batch_data_generation_WV(name, c_min_WV, c_max_WV, database_file):
	
	database = pd.read_csv(database_file, sep = ";", header = None)
	with open('./data/df_mesocyclones.p', 'rb') as fp:
		df = pickle.load(fp)
	
	#download the whole track for a given cyclone's name
	pi_name = "./data/Cuts_Mesocyclones/" + str(name)
	with open(pi_name, 'rb') as fp:
		cyclone_track = pickle.load(fp)

	#normalize each array between 0 and 1
	cyclone_track_data = [cyclone_track[1][i][np.newaxis,:,:] for i in range(len(cyclone_track[1]))]
	cyclone_track_data = np.concatenate(cyclone_track_data, axis=0)
	cyclone_track_data_scaled = (cyclone_track_data-c_min_WV)/(c_max_WV-c_min_WV) 
	#filtering
	cyclone_track_data_scaled.ravel()[cyclone_track_data_scaled.ravel() < 0] = 0

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
	for i in range(0,int(len(cyclone_track_data_scaled))-1,1):
		
		database_srez = database[database[2] == name]
		time_next = datetime.strptime(str(int(database_srez[3].iloc[i+1])), '%Y%m%d%H')
		ex = df.loc[df['datetime'] == time_next]
		for m in ex['name']:
			if m != name:
				local_name = "./data/Cuts_Mesocyclones/" + str(m)
				with open(local_name, 'rb') as ln:
					local_track = pickle.load(ln)

					#normilize local_track
					cyclone_local_data = [local_track[1][k][np.newaxis,:,:] for k in range(len(local_track[1]))]
					cyclone_local_data = np.concatenate(cyclone_local_data, axis=0)
					cyclone_local_data_scaled_BLYAT = (cyclone_local_data-c_min_WV)/(c_max_WV-c_min_WV) 
					#filtering
					cyclone_local_data_scaled_BLYAT.ravel()[cyclone_local_data_scaled_BLYAT.ravel() < 0] = 0
					try:
						X_1.append(cyclone_track_data_scaled[i])
						X_2.append(cyclone_local_data_scaled_BLYAT[int(ex[ex['name']== m]['index'])])
						y.append(0)
						break
					except:
						pass


	return np.asarray(X_1), np.asarray(X_2), np.asarray(y)
	
def MCNND_augmentation(database_file, train_names, validation_names, test_names, c_min_WV, c_max_WV, c_min_IR, c_max_IR):
	seq = iaa.Sequential([
		iaa.Affine(
				rotate=(-5.1, 5.1), # rotate by -45 to +45 degrees
				shear=(-1.1, 1.1), # shear by -16 to +16 degrees
			),
		iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
		iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
		iaa.Sharpen(alpha=(0, 0.03), lightness=(0.75, 1.5)) # sharpen images
	])
	
	for name in tqdm(train_names[:]):
		
		X_1_WV,X_2_WV, y = MCNND_batch_data_generation_WV(name, c_min_WV, c_max_WV, database_file)
		X_1_IR,X_2_IR,y, dist = MCNND_batch_data_generation_IR(name, c_min_IR, c_max_IR, database_file)
				
		#AUGMENTATION
		
		for i in range(20):
			aug_list = []
			
			seq_det = seq.to_deterministic()
					
			X_1_WV_aug = np.stack((seq_det.augment_images(X_1_WV*255),)*3, axis=-1)
			X_2_WV_aug = np.stack((seq_det.augment_images(X_2_WV*255),)*3, axis=-1)
			X_1_IR_aug = np.stack((seq_det.augment_images(X_1_IR*255),)*3, axis=-1)
			X_2_IR_aug = np.stack((seq_det.augment_images(X_2_IR*255),)*3, axis=-1)
			
			aug_list.append(X_1_WV_aug)
			aug_list.append(X_2_WV_aug)
			aug_list.append(X_1_IR_aug)
			aug_list.append(X_2_IR_aug)
			aug_list.append(y)
			aug_list.append(dist)
			
			pi_name = "./data/datasets/train_dist/" + str(name) + "_" + str(i) 
					
			with open(pi_name, 'wb') as fp:
				pickle.dump(aug_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
	
	for name in tqdm(validation_names[:]):
		aug_list = []
		
		X_1_WV,X_2_WV,y = MCNND_batch_data_generation_WV(name, c_min_WV, c_max_WV, database_file)
		X_1_IR,X_2_IR,y,dist = MCNND_batch_data_generation_IR(name, c_min_IR, c_max_IR, database_file)
		
		X_1_WV = np.stack((X_1_WV*255,)*3, axis=-1)
		X_2_WV = np.stack((X_2_WV*255,)*3, axis=-1)
		X_1_IR = np.stack((X_1_IR*255,)*3, axis=-1)
		X_2_IR = np.stack((X_2_IR*255,)*3, axis=-1)    
				
		aug_list.append(X_1_WV)
		aug_list.append(X_2_WV)
		aug_list.append(X_1_IR)
		aug_list.append(X_2_IR)
		aug_list.append(y)
		aug_list.append(dist)
			
		pi_name = "./data/datasets/validation_dist/" + str(name)
					
		with open(pi_name, 'wb') as fp:
			pickle.dump(aug_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
	
	for name in tqdm(test_names[:]):
		aug_list = []
		
		X_1_WV,X_2_WV,y = MCNND_batch_data_generation_WV(name, c_min_WV, c_max_WV, database_file)
		X_1_IR,X_2_IR,y, dist = MCNND_batch_data_generation_IR(name, c_min_IR, c_max_IR, database_file)
		
		X_1_WV = np.stack((X_1_WV*255,)*3, axis=-1)
		X_2_WV = np.stack((X_2_WV*255,)*3, axis=-1)
		X_1_IR = np.stack((X_1_IR*255,)*3, axis=-1)
		X_2_IR = np.stack((X_2_IR*255,)*3, axis=-1)  
				
		aug_list.append(X_1_WV)
		aug_list.append(X_2_WV)
		aug_list.append(X_1_IR)
		aug_list.append(X_2_IR)
		aug_list.append(y)
		aug_list.append(dist)
			
		pi_name = "./data/datasets/test_dist/" + str(name)
					
		with open(pi_name, 'wb') as fp:
			pickle.dump(aug_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
			
def MCNN_augmentation(database_file, train_names, validation_names, test_names, c_min_WV, c_max_WV, c_min_IR, c_max_IR):
	seq = iaa.Sequential([
		iaa.Affine(
				rotate=(-5.1, 5.1), # rotate by -45 to +45 degrees
				shear=(-1.1, 1.1), # shear by -16 to +16 degrees
			),
		iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
		iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
		iaa.Sharpen(alpha=(0, 0.03), lightness=(0.75, 1.5)) # sharpen images
	])
	
	for name in tqdm(train_names[:]):
		
		X_1_WV,X_2_WV, y = MCNN_batch_data_generation_WV(name, c_min_WV, c_max_WV, database_file)
		X_1_IR,X_2_IR,y = MCNN_batch_data_generation_IR(name, c_min_IR, c_max_IR, database_file)
				
		#AUGMENTATION
		
		for i in range(20):
			aug_list = []
			
			seq_det = seq.to_deterministic()
					
			X_1_WV_aug = np.stack((seq_det.augment_images(X_1_WV*255),)*3, axis=-1)
			X_2_WV_aug = np.stack((seq_det.augment_images(X_2_WV*255),)*3, axis=-1)
			X_1_IR_aug = np.stack((seq_det.augment_images(X_1_IR*255),)*3, axis=-1)
			X_2_IR_aug = np.stack((seq_det.augment_images(X_2_IR*255),)*3, axis=-1)
			
			aug_list.append(X_1_WV_aug)
			aug_list.append(X_2_WV_aug)
			aug_list.append(X_1_IR_aug)
			aug_list.append(X_2_IR_aug)
			aug_list.append(y)
			
			pi_name = "./data/datasets/train/" + str(name) + "_" + str(i) 
					
			with open(pi_name, 'wb') as fp:
				pickle.dump(aug_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
	
	for name in tqdm(validation_names[:]):
		aug_list = []
		
		X_1_WV,X_2_WV,y = MCNN_batch_data_generation_WV(name, c_min_WV, c_max_WV, database_file)
		X_1_IR,X_2_IR,y = MCNN_batch_data_generation_IR(name,c_min_IR, c_max_IR, database_file)
		
		X_1_WV = np.stack((X_1_WV*255,)*3, axis=-1)
		X_2_WV = np.stack((X_2_WV*255,)*3, axis=-1)
		X_1_IR = np.stack((X_1_IR*255,)*3, axis=-1)
		X_2_IR = np.stack((X_2_IR*255,)*3, axis=-1)    
				
		aug_list.append(X_1_WV)
		aug_list.append(X_2_WV)
		aug_list.append(X_1_IR)
		aug_list.append(X_2_IR)
		aug_list.append(y)
			
		pi_name = "./data/datasets/validation/" + str(name)
					
		with open(pi_name, 'wb') as fp:
			pickle.dump(aug_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
	
	for name in tqdm(test_names[:]):
		aug_list = []
		
		X_1_WV,X_2_WV,y = MCNN_batch_data_generation_WV(name, c_min_WV, c_max_WV, database_file)
		X_1_IR,X_2_IR,y = MCNN_batch_data_generation_IR(name, c_min_IR, c_max_IR, database_file)
		
		X_1_WV = np.stack((X_1_WV*255,)*3, axis=-1)
		X_2_WV = np.stack((X_2_WV*255,)*3, axis=-1)
		X_1_IR = np.stack((X_1_IR*255,)*3, axis=-1)
		X_2_IR = np.stack((X_2_IR*255,)*3, axis=-1)  
				
		aug_list.append(X_1_WV)
		aug_list.append(X_2_WV)
		aug_list.append(X_1_IR)
		aug_list.append(X_2_IR)
		aug_list.append(y)
			
		pi_name = "./data/datasets/test/" + str(name)
					
		with open(pi_name, 'wb') as fp:
			pickle.dump(aug_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
	