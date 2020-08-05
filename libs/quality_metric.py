import numpy as np
import pickle
from tensorflow.keras.applications.vgg19 import preprocess_input
from tqdm import tqdm as tqdm_base
from math import sin, cos, sqrt, atan2, radians

def tqdm(*args, **kwargs):
	if hasattr(tqdm_base, '_instances'):
		for instance in list(tqdm_base._instances):
			tqdm_base._decr_instances(instance)
	return tqdm_base(*args, **kwargs)


def MOTA(predicted_tracks, ground_truth_tracks, df, datetimes):
	global_mota = []
	
	for key in tqdm(predicted_tracks.keys()):

		motas_for_given_track = []
		
		track = predicted_tracks[key]
		
		start_time = str(df[(df['name'] == track[0][0]) & (df['index'] == track[0][1])]['datetime'].tolist()[0])
		end_time = str(df[(df['name'] == track[-1][0]) & (df['index'] == track[-1][1])]['datetime'].tolist()[0])
		
		for start_index,j in enumerate(datetimes):
			if j == np.datetime64(start_time):
				break

		for end_index,j in enumerate(datetimes):
			if j == np.datetime64(end_time): 
				break
				
		indexes = [i for i in range(start_index, end_index + 1)]
				
		alls = []
		for i in range(len(datetimes)):
			if ((datetimes[i] >= np.datetime64(start_time)) &  (datetimes[i] <= np.datetime64(end_time))):
				sr = df[df['datetime'] == datetimes[i]]['name'].tolist()
				for j in sr:
					if j not in alls:
						alls.append(j)
					
		for true_name in alls:
			start_true = str(df[(df['name'] == ground_truth_tracks[true_name][0][0]) & (df['index'] == ground_truth_tracks[true_name][0][1])]['datetime'].tolist()[0])
			if np.datetime64(start_true) not in datetimes:
				start_true = "2004-09-07 03:00:00"
				for i in range(len(ground_truth_tracks[true_name])):
					shift = str(df[(df['name'] == ground_truth_tracks[true_name][i][0]) & (df['index'] == ground_truth_tracks[true_name][i][1])]['datetime'].tolist()[0])
					if shift == start_true:
						ground_truth_tracks[true_name] = ground_truth_tracks[true_name][i:]
						break
				
				
				
			end_true = str(df[(df['name'] == ground_truth_tracks[true_name][-1][0]) & (df['index'] == ground_truth_tracks[true_name][-1][1])]['datetime'].tolist()[0])
			
			try:
				for start_index_true,j in enumerate(datetimes):
					if j == np.datetime64(start_true):
						break

				for end_index_true,j in enumerate(datetimes):
					if j == np.datetime64(end_true):
						break
				
				indexes_true = [i for i in range(start_index_true, end_index_true + 1)]
				if len(indexes_true) == 0:
					continue
				commons, pred_inds, tr_inds = np.intersect1d(indexes,indexes_true, return_indices=True)
				pred_ind = pred_inds[0]
				tr_ind = tr_inds[0]

			except:
				continue
			
			# False Positives computing
			try:
				FP = 0
				for _ in range(len(commons)):
					if track[pred_inds[pred_ind]][2:] != ground_truth_tracks[true_name][tr_inds[tr_ind]][2:]:      
						FP += 1
					pred_ind += 1 
					tr_ind += 1
					
			except:
				
				FP = 0
						
			
			# Misses computing       
			if np.datetime64(end_time) > np.datetime64(end_true):
				Miss = pred_inds[0]
			else:
				Miss = pred_inds[0] + len(indexes_true) - len(indexes)
	
			mota = 1 - (FP + Miss)/len(ground_truth_tracks[true_name])
			
			motas_for_given_track.append(mota)
		
		mean_mota = np.mean(motas_for_given_track)
		global_mota.append(mean_mota)
	
	return np.mean(global_mota)
	

def calculate_distance(lat1, lon1,lat2, lon2):
	R = 6373.0
	lat1 = radians(lat1)
	lon1 = radians(lon1)
	lat2 = radians(lat2)
	lon2 = radians(lon2)

	dlon = lon2 - lon1
	dlat = lat2 - lat1

	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))

	distance = R * c
	return distance



def MC_open_image_and_filtering_prerocess_for_NN(name, index):
	
	pi_name = "./data/Cuts_Mesocyclones/" + str(name)
	c_min_IR = 230.0
	c_max_IR = 330.0
	c_min_WV = 230.0
	c_max_WV = 330.0
	
	with open(pi_name, 'rb') as fp:

		cyclone_track = pickle.load(fp)
		cyclone_track_WV = cyclone_track[1][index]
		cyclone_track_IR = cyclone_track[0][index]

		WV_cyclone_track_data_scaled = (cyclone_track_WV - c_min_IR)/(c_max_IR-c_min_IR) 
		#filtering WV
		WV_cyclone_track_data_scaled.ravel()[WV_cyclone_track_data_scaled.ravel() < 0] = 0

		IR_cyclone_track_data_scaled = (cyclone_track_IR-c_min_IR)/(c_max_IR-c_min_IR) 
		#filtering IR
		IR_cyclone_track_data_scaled.ravel()[IR_cyclone_track_data_scaled.ravel() < 0] = 0
		
		WV_now_work_with = np.copy(IR_cyclone_track_data_scaled)
		IR_now_work_with = np.copy(IR_cyclone_track_data_scaled)


		WV_now_work_with_prep = preprocess_input(np.stack((WV_now_work_with*255,)*3, axis=-1))[:,:,0]
		IR_now_work_with_prep = preprocess_input(np.stack((IR_now_work_with*255,)*3, axis=-1))[:,:,0]


		WV_now_work_with_prep = WV_now_work_with_prep.reshape(1,200,200,1)
		IR_now_work_with_prep = IR_now_work_with_prep.reshape(1,200,200,1)


		concat_X1 = np.concatenate([IR_now_work_with_prep, WV_now_work_with_prep], axis=3)
		

		
		
	return concat_X1
	
def TC_open_image_and_filtering_prerocess_for_NN(name, index):
	c_min = 92352.9
	c_max = 105111.2
	
	pi_name = "./data/Cuts/" + str(name)
	
	with open(pi_name, 'rb') as fp:
		cyclone_track = pickle.load(fp)
		cyclone_track = cyclone_track[index][0]
		cyclone_track  = [cyclone_track [np.newaxis,:,:]]
		#cyclone_track = np.concatenate(cyclone_track_data, axis=0)
		cyclone_track = (cyclone_track[0]-c_min)/(c_max-c_min)
		
		expanded = np.asarray([np.expand_dims(x,-1) for x in cyclone_track])
		
		
	return expanded