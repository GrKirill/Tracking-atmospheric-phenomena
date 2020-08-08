import pickle
import imgaug as ia
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta
import os
from libs.service import *
from tqdm import tqdm as tqdm_base
try:
	import cPickle as pickle
except ImportError: 
	import pickle

def tqdm(*args, **kwargs):
	if hasattr(tqdm_base, '_instances'):
		for instance in list(tqdm_base._instances):
			tqdm_base._decr_instances(instance)
	return tqdm_base(*args, **kwargs)

def main(args=None):
	if args is None:
		args = sys.argv[1:]
			
	database_file = str(args[0])
	path_to_slp = str(args[1])
	
	data_dir = os.path.join(os.path.abspath('./'), 'data')
	try:
		EnsureDirectoryExists(data_dir)
	except:
		print('data directory couldn`t be found and couldn`t be created:\n%s' % data_dir)
		raise FileNotFoundError('data directory couldn`t be found and couldn`t be created:\n%s' % data_dir)
		
	cuts_dir = os.path.join(os.path.abspath('./data/'), 'Cuts')
	try:
		EnsureDirectoryExists(cuts_dir)
	except:
		print('cuts directory couldn`t be found and couldn`t be created:\n%s' % cuts_dir)
		raise FileNotFoundError('cuts directory couldn`t be found and couldn`t be created:\n%s' % cuts_dir)
	
	cuts(database_file, path_to_slp)
	
	
def cuts(database_file, path_to_slp):
	
	hudrat = pd.read_csv(database_file, delimiter = ",", skiprows = [1])
	hudrat['ISO_TIME'] = pd.to_datetime(hudrat['ISO_TIME'])
	cyclones = {}
	for i in range(hudrat.shape[0]):
		if hudrat['SID'][i] in cyclones:
			cyclones[hudrat['SID'][i]].append([hudrat['ISO_TIME'][i], hudrat['LAT'][i], hudrat['LON'][i]])
		else:
			cyclones[hudrat['SID'][i]] = []  
	cyclones = dict( [(k,v) for k,v in cyclones.items() if len(v)>0])
	
	l = datetime(1970, 1, 31, 23, 0)
	k = datetime(2017, 1, 31, 23, 0)
	cyclones_20_years= []
	for i in cyclones.keys():
		if cyclones[i][0][0]>= l:
			if cyclones[i][-1][0] <= k:
				cyclones_20_years.append(i)
				
	iterator = 0
	for name in tqdm(cyclones_20_years[:]):
		dict_for_arrays = {}
		
		num_of_tracks = len(cyclones[name])
		
		iterator+=1
		for i in range(num_of_tracks):
			datetime_to_find = cyclones[name][i][0]
			year = datetime_to_find.year
			month = datetime_to_find.month
			if len(str(month)) == 2:
				month = month
			else:
				month = '0' + str(month)
			dataset_name =  path_to_slp + "era5_mslp_" + str(year)+ '-' + str(month) + '.nc'#/storage/tartar/DATA/ERA5/slp/
			mlp_ds = Dataset(dataset_name, 'r')
			lats = np.copy(mlp_ds.variables['latitude'])
			lons = np.copy(mlp_ds.variables['longitude'])
			lons_mesh,lats_mesh = np.meshgrid(lons,lats)
		
			dt_ancor = datetime(1900,1,1)
			ds_datetimes = np.asarray([dt_ancor + timedelta(hours=int(dt)) for dt in mlp_ds['time'][:].data])
			try:
				idx = np.where(ds_datetimes == datetime_to_find)[0][0]
				mslp_current = np.copy(mlp_ds['msl'][idx])
				#close dataset
				mlp_ds.close()
				rho = np.sqrt((lats_mesh-cyclones[name][i][1])**2 + (lons_mesh-cyclones[name][i][2])**2)
				center_pt = np.unravel_index(np.argmin(rho), rho.shape)
				cut = mslp_current[center_pt[0]-40:center_pt[0]+40, center_pt[1]-40:center_pt[1]+40]
				if all(i > 0 for i in center_pt):
					key = name
					dict_for_arrays.setdefault(key, [])
					dict_for_arrays[key].append([np.asarray(cut),ds_datetimes[i]])	   
			except:
				mlp_ds.close()
				pass
					
		pi_name = "./data/Cuts/" + name
				
		if len(dict_for_arrays) != 0:
			with open(pi_name, 'wb') as fp:
				pickle.dump(dict_for_arrays[name] , fp, protocol=pickle.HIGHEST_PROTOCOL)
				
				
if __name__ == '__main__':
	main()