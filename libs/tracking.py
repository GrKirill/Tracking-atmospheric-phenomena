import numpy as np
from tqdm import tqdm as tqdm_base
from libs.quality_metric import *


def tqdm(*args, **kwargs):
	if hasattr(tqdm_base, '_instances'):
		for instance in list(tqdm_base._instances):
			tqdm_base._decr_instances(instance)
	return tqdm_base(*args, **kwargs)
	
def tracking_cyclones(df, model_name):
	
	datetimes = df['datetime'].values
	datetimes = np.unique(datetimes)
	datetimes.sort()
	datetimes = list(datetimes)
	date_start = "2010-01-01 15:00:00"
	for start_index,j in enumerate(datetimes):
		if j == np.datetime64(date_start):
			break
			
	datetimes = datetimes[start_index:] 	
	global_dict = {}

	#initial dict
	initial_srez = df[df['datetime'] == datetimes[0]]
	for i in range(len(initial_srez)):
		global_dict[i] = []
		global_dict[i].append(list(initial_srez.values[i][1:]))
		
	if model_name == "tcnn":
			
		from libs.models import TCNN, batch_data_generation
		sim_model = TCNN().get_siamese_model()
		sim_model.load_weights("./logs/tcnn/tcnn.h5")
		for key in tqdm(global_dict.keys()):
			for timestamp in range(0,300):
				name,index,lat,long = global_dict[key][-1]
				first_input = TC_open_image_and_filtering_prerocess_for_NN(name,index)
				next_index = timestamp + 1
				probs = {}
				for new_name, new_index, new_lat, new_long in df[df['datetime'] == datetimes[next_index]][['name', 'index', 'lat', 'long']].values: 
					second_input = TC_open_image_and_filtering_prerocess_for_NN(new_name, new_index)
					proba = sim_model.predict_on_batch([first_input, second_input])
					probs[new_name] = [proba, new_index,  new_lat, new_long]
				target_name = max(probs.items(), key=lambda x: x[1][0])[0]
				target_index, target_lat, target_long = probs[target_name][1:]
				
				if probs[target_name][0] < 0.15 :
					break
					
				global_dict[key].append([target_name, target_index, target_lat, target_long])
	
	if model_name == "random_tc":
		
		for key in tqdm(global_dict.keys()):
			for timestamp in range(0,300):
				#open what we want to compare
				name,index,lat,long = global_dict[key][-1]
				next_index = timestamp + 1
				probs = {}
				for new_name, new_index, new_lat, new_long in df[df['datetime'] == datetimes[next_index]][['name', 'index', 'lat', 'long']].values:
					proba = np.random.uniform(0, 1)
					probs[new_name] = [proba, new_index,  new_lat, new_long]
				target_name = max(probs.items(), key=lambda x: x[1][0])[0]
				target_index, target_lat, target_long = probs[target_name][1:]
			
				if probs[target_name][0] < 0.5 :
					break
					
				global_dict[key].append([target_name, target_index, target_lat, target_long])
#---------------------------------------------------------------------------------------------------------------------------------------------------
	if model_name == "tcnn":
				
		for global_timestamp in tqdm(range(1, len(datetimes)-1)):
			current_srez = df[df['datetime'] == datetimes[global_timestamp]]
			new_keys = []
			for find_new in current_srez.values:
				isNew = True
				for key in global_dict.keys():
					for item in global_dict[key]:
						if (item[2] == find_new[3]) & (item[3] == find_new[4]):
							isNew = False
							break
				if isNew:
					key_to_add = len(global_dict)
					new_keys.append(key_to_add)
					global_dict[key_to_add] = []
					global_dict[key_to_add].append(list(find_new[1:]))

			for new_key in new_keys:
				max_time_stamp = global_timestamp + 300
				if max_time_stamp >= len(datetimes)-1:
					max_time_stamp = len(datetimes)-2
				
				for timestamp in range(global_timestamp, max_time_stamp):
					name,index,lat,long = global_dict[new_key][-1]
					
					try:
						first_input = TC_open_image_and_filtering_prerocess_for_NN(name,index)
					except:
						break

					next_index = timestamp + 1
					probs = {}
					for new_name, new_index, new_lat, new_long in df[df['datetime'] == datetimes[next_index]][['name', 'index', 'lat', 'long']].values: 

						try:
							second_input = TC_open_image_and_filtering_prerocess_for_NN(new_name, new_index)
						except:
							continue

						proba = sim_model.predict_on_batch([first_input, second_input])
						probs[new_name] = [proba, new_index,  new_lat, new_long]

					try:
						target_name = max(probs.items(), key=lambda x: x[1][0])[0]
						target_index, target_lat, target_long = probs[target_name][1:]
						
						if probs[target_name][0] < 0.15:
							break
							
						global_dict[new_key].append([target_name, target_index, target_lat, target_long])

					except:
						break  
							
	if model_name == "random_tc":
		for global_timestamp in tqdm(range(1, len(datetimes)-1)):
			current_srez = df[df['datetime'] == datetimes[global_timestamp]]
			new_keys = []
			for find_new in current_srez.values:
				isNew = True
				for key in global_dict.keys():
					for item in global_dict[key]:
						if (item[2] == find_new[3]) & (item[3] == find_new[4]):
							isNew = False
							break
				if isNew:
					key_to_add = len(global_dict)
					new_keys.append(key_to_add)
					global_dict[key_to_add] = []
					global_dict[key_to_add].append(list(find_new[1:]))

			for new_key in new_keys:
				max_time_stamp = global_timestamp + 300
				if max_time_stamp >= len(datetimes)-1:
					max_time_stamp = len(datetimes)-2
				
				for timestamp in range(global_timestamp, max_time_stamp):
					#open what we want to compare
					name,index,lat,long = global_dict[new_key][-1]
					next_index = timestamp + 1
					probs = {}
					for new_name, new_index, new_lat, new_long in df[df['datetime'] == datetimes[next_index]][['name', 'index', 'lat', 'long']].values:	
						proba = np.random.uniform(0, 1)
						probs[new_name] = [proba, new_index,  new_lat, new_long]

					try:
						target_name = max(probs.items(), key=lambda x: x[1][0])[0]
						target_index, target_lat, target_long = probs[target_name][1:]
						
						if probs[target_name][0] < 0.5:
							break
							
						global_dict[new_key].append([target_name, target_index, target_lat, target_long])
					except:
						break  

						
	return global_dict, datetimes
	

def tracking_mesocyclones(df, model_name):
		
	datetimes = df['datetime'].values
	datetimes = np.unique(datetimes)
	datetimes.sort()
	datetimes = list(datetimes)
	date_start = "2004-09-07 03:00:00"
	for start_index,j in enumerate(datetimes):
		if j == np.datetime64(date_start):
			break
			
	datetimes = datetimes[start_index:] 		
	global_dict = {}
	
	#initial dict
	initial_srez = df[df['datetime'] == datetimes[0]]
	for i in range(len(initial_srez)):
		global_dict[i] = []
		global_dict[i].append(list(initial_srez.values[i][1:]))
	
	if model_name == "mcnn":
		#load weights
		from libs.models import MCNN
		sim_model = MCNN().get_siamese_model()
		sim_model.load_weights("./pretrained_models/mcnn.h5")
	
	if model_name == "mcnnd":
		#load weights
		from libs.models import MCNNd
		sim_model = MCNNd().get_siamese_model()
		sim_model.load_weights("./pretrained_models/mcnnd.h5")#pretrained
		
	if (model_name == 'mcnn') or (model_name == 'mcnnd'):
		for key in tqdm(global_dict.keys()):
				for timestamp in range(0,100):
					name,index,lat,long = global_dict[key][-1]
						
					try:
						first_input = MC_open_image_and_filtering_prerocess_for_NN(name,index)
					except:
						break
							
					next_index = timestamp + 1
					probs = {}
					for new_name, new_index, new_lat, new_long in df[df['datetime'] == datetimes[next_index]][['name', 'index', 'lat', 'long']].values: 
							
						try:
							second_input = MC_open_image_and_filtering_prerocess_for_NN(new_name, new_index)
						except:
							break
							
						if model_name == "mcnn":
							proba = sim_model.predict_on_batch([first_input, second_input])
							
						if model_name == "mcnnd":
							distance = np.array(calculate_distance(lat, long, new_lat, new_long)).reshape(-1,1)
							proba = sim_model.predict_on_batch([first_input, second_input, distance])
							
						probs[new_name] = [proba, new_index,  new_lat, new_long]
						
					try:
						target_name = max(probs.items(), key=lambda x: x[1][0])[0]
						target_index, target_lat, target_long = probs[target_name][1:]
						if probs[target_name][0] < 0.9 :
							break
								
						global_dict[key].append([target_name, target_index, target_lat, target_long])	
					except:
						break		
					
	if model_name == "random_mc":
		for key in tqdm(global_dict.keys()):
			for timestamp in range(0,100):
				name,index,lat,long = global_dict[key][-1]
				next_index = timestamp + 1
				probs = {}
				for new_name, new_index, new_lat, new_long in df[df['datetime'] == datetimes[next_index]][['name', 'index', 'lat', 'long']].values: 
					proba = np.random.uniform(0, 1)
					probs[new_name] = [proba, new_index,  new_lat, new_long]
						
				try:
					target_name = max(probs.items(), key=lambda x: x[1][0])[0]
					target_index, target_lat, target_long = probs[target_name][1:]
					if probs[target_name][0] < 0.5:
						break
					
					global_dict[key].append([target_name, target_index, target_lat, target_long])	
				except:
					break
	
			
	#----------------------------------------------------------------------------------------------------------------

	if (model_name == 'mcnn') or (model_name == 'mcnnd'):
		for global_timestamp in tqdm(range(1, len(datetimes)-1)):
			current_srez = df[df['datetime'] == datetimes[global_timestamp]]
			new_keys = []
			for find_new in current_srez.values:
				isNew = True
				for key in global_dict.keys():
					for item in global_dict[key]:
						if (item[2] == find_new[3]) & (item[3] == find_new[4]):
							isNew = False
							break
				if isNew:
					key_to_add = len(global_dict)
					new_keys.append(key_to_add)
					global_dict[key_to_add] = []
					global_dict[key_to_add].append(list(find_new[1:]))

			for new_key in new_keys:
				
				max_time_stamp = global_timestamp + 100
				if max_time_stamp >= len(datetimes)-1:
					max_time_stamp = len(datetimes)-2
					
				if (model_name == "mcnn") or (model_name == "mcnnd"):
					for timestamp in range(global_timestamp, max_time_stamp):
						name,index,lat,long = global_dict[new_key][-1]
						
						try:
							first_input = MC_open_image_and_filtering_prerocess_for_NN(name,index)
						except:
							break

						next_index = timestamp + 1
						probs = {}
						for new_name, new_index, new_lat, new_long in df[df['datetime'] == datetimes[next_index]][['name', 'index', 'lat', 'long']].values: 

							try:
								second_input = MC_open_image_and_filtering_prerocess_for_NN(new_name, new_index)
							except:
								continue

							if model_name == "mcnn":
								proba = sim_model.predict_on_batch([first_input, second_input])
							
							if model_name == "mcnnd":
								distance = np.array(calculate_distance(lat, long, new_lat, new_long)).reshape(-1,1)
								proba = sim_model.predict_on_batch([first_input, second_input, distance])
								
							probs[new_name] = [proba, new_index,  new_lat, new_long]

						try:
							target_name = max(probs.items(), key=lambda x: x[1][0])[0]
							target_index, target_lat, target_long = probs[target_name][1:]
							if probs[target_name][0] < 0.9:
								break
									
							global_dict[new_key].append([target_name, target_index, target_lat, target_long])
						except:
							break  			
						
		if model_name == "random_mc":
			for timestamp in range(global_timestamp, max_time_stamp):
				name,index,lat,long = global_dict[new_key][-1]
				next_index = timestamp + 1
				probs = {}
				for new_name, new_index, new_lat, new_long in df[df['datetime'] == datetimes[next_index]][['name', 'index', 'lat', 'long']].values: 
					proba = np.random.uniform(0, 1)
					probs[new_name] = [proba, new_index,  new_lat, new_long]

				try:
					target_name = max(probs.items(), key=lambda x: x[1][0])[0]
					target_index, target_lat, target_long = probs[target_name][1:]
					if probs[target_name][0] < 0.5:
						break
							
					global_dict[new_key].append([target_name, target_index, target_lat, target_long])
				except:
					break    

							
	return global_dict, datetimes
			