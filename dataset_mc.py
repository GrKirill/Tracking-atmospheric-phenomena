import pickle
import imgaug as ia
import numpy as np
import pandas as pd
import os
from libs.service import *
from libs.quality_metric import calculate_distance
from tqdm import tqdm as tqdm_base
from libs.mesocyclones_gen import *

def tqdm(*args, **kwargs):
	if hasattr(tqdm_base, '_instances'):
		for instance in list(tqdm_base._instances):
			tqdm_base._decr_instances(instance)
	return tqdm_base(*args, **kwargs)


def main(args=None):
	if args is None:
		args = sys.argv[1:]
		
	database_file = str(args[0])
	path_to_mosaics = str(args[1])
	
	data_dir = os.path.join(os.path.abspath('./'), 'data')
	try:
		EnsureDirectoryExists(data_dir)
	except:
		print('data directory couldn`t be found and couldn`t be created:\n%s' % data_dir)
		raise FileNotFoundError('data directory couldn`t be found and couldn`t be created:\n%s' % data_dir)
	
	datasets_dir = os.path.join(os.path.abspath('./data/'), 'datasets')
	try:
		EnsureDirectoryExists(data_dir)
	except:
		print('datasets directory couldn`t be found and couldn`t be created:\n%s' % datasets_dir)
		raise FileNotFoundError('datasets directory couldn`t be found and couldn`t be created:\n%s' % datasets_dir)
	
	cuts_mc_dir = os.path.join(os.path.abspath('./data/'), 'Cuts_Mesocyclones')
	try:
		EnsureDirectoryExists(cuts_mc_dir)
	except:
		print('cuts_mc directory couldn`t be found and couldn`t be created:\n%s' % cuts_mc_dir)
		raise FileNotFoundError('cuts_mc directory couldn`t be found and couldn`t be created:\n%s' % cuts_mc_dir)
		
	train_dir = os.path.join(os.path.abspath('./data/datasets/'), 'train')
	try:
		EnsureDirectoryExists(train_dir )
	except:
		print('train directory couldn`t be found and couldn`t be created:\n%s' % train_dir )
		raise FileNotFoundError('train directory couldn`t be found and couldn`t be created:\n%s' % train_dir )
	
	test_dir = os.path.join(os.path.abspath('./data/datasets/'), 'test')
	try:
		EnsureDirectoryExists(test_dir)
	except:
		print('test directory couldn`t be found and couldn`t be created:\n%s' % test_dir )
		raise FileNotFoundError('test directory couldn`t be found and couldn`t be created:\n%s' % test_dir)
	
	validation_dir = os.path.join(os.path.abspath('./data/datasets/'), 'validation')
	try:
		EnsureDirectoryExists(validation_dir)
	except:
		print('validation directory couldn`t be found and couldn`t be created:\n%s' % validation_dir)
		raise FileNotFoundError('validation directory couldn`t be found and couldn`t be created:\n%s' % validation_dir)
	
	train_dist_dir = os.path.join(os.path.abspath('./data/datasets/'), 'train_dist')
	try:
		EnsureDirectoryExists(train_dist_dir)
	except:
		print('train_dist directory couldn`t be found and couldn`t be created:\n%s' % train_dist_dir )
		raise FileNotFoundError('train directory couldn`t be found and couldn`t be created:\n%s' %train_dist_dir )
	
	validation_dist_dir = os.path.join(os.path.abspath('./data/datasets/'), 'validation_dist')
	try:
		EnsureDirectoryExists(validation_dist_dir)
	except:
		print('validation_dist directory couldn`t be found and couldn`t be created:\n%s' % validation_dist_dir)
		raise FileNotFoundError('train directory couldn`t be found and couldn`t be created:\n%s' %validation_dist_dir)
	
	test_dist_dir = os.path.join(os.path.abspath('./data/datasets/'), 'test_dist')
	try:
		EnsureDirectoryExists(test_dist_dir)
	except:
		print('test_dist directory couldn`t be found and couldn`t be created:\n%s' % test_dist_dir)
		raise FileNotFoundError('train directory couldn`t be found and couldn`t be created:\n%s' %test_dist_dir)
	
	c_min_IR, c_min_WV = 230, 230
	c_max_IR, c_max_WV = 330, 330
	names = list(range(1,1736))
	train_names = list(range(1,1201))
	validation_names = list(range(1201,1401))
	test_names = list(range(1401, 1736))
	df(database_file, names)
	create_cuts(database_file, path_to_mosaics)
	MCNND_augmentation(database_file, train_names, validation_names, test_names, c_min_WV, c_max_WV, c_min_IR, c_max_IR)
	MCNN_augmentation(database_file, train_names, validation_names, test_names, c_min_WV, c_max_WV, c_min_IR, c_max_IR)
	

if __name__ == '__main__':
	main()
	
	
