from libs.parse_args import *
from libs.quality_metric import *
from libs.tracking import *

import sys

def main(args=None):
	if args is None:
		args = sys.argv[1:]
		
	model_name = str(args[0]) #mcnn mcnnd tcnn random_mc
	if (model_name == 'mcnnd') or (model_name == 'mcnn') or (model_name == 'random_mc'):
		with open('./data/df_mesocyclones.p', 'rb') as fp:
			df = pickle.load(fp)
			global_dict, datetimes = tracking_mesocyclones(df, model_name)
	if (model_name == 'tcnn') or (model_name == 'random_tc'):
		with open('./data/df_tropical_cyclones.p', 'rb') as fp:
			df = pickle.load(fp)
			global_dict, datetimes = tracking_cyclones(df, model_name)
												

	names_true_tracks = np.unique([track[i][0] for track in list(global_dict.values()) for i in range(len(track))])
	true_tracks_dict = {}
	for name in names_true_tracks:
		true_tracks_dict[name] = df[df['name'] == name].values[:,1:].tolist()
		
	mota_value = MOTA(global_dict, true_tracks_dict, df, datetimes)
	print('MOTA value is:', mota_value)
		
	
if __name__ == '__main__':
	main()
