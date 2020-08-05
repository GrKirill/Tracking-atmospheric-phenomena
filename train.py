from libs.parse_args import *
import sys
from libs.service import *
import pathlib

def main(args=None):
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)
	
	model_name = args.model
	epochs = args.epochs
	learning_rate = args.learning_rate
	input_shape = eval(args.input_shape)
	
	logs_dir = os.path.join(os.path.abspath('./'), 'logs')
	try:
		EnsureDirectoryExists(logs_dir)
	except:
		print('logs directory couldn`t be found and couldn`t be created:\n%s' % logs_dir)
		raise FileNotFoundError('datasets directory couldn`t be found and couldn`t be created:\n%s' % logs_dir)
		
	data_dir = logs_dir = os.path.join(os.path.abspath('./'), 'data')
	try:
		EnsureDirectoryExists(data_dir)
	except:
		print('data directory couldn`t be found and couldn`t be created:\n%s' % data_dir)
		raise FileNotFoundError('data directory couldn`t be found and couldn`t be created:\n%s' % data_dir)
	
	if model_name == 'mcnn':
		pathlib.Path('./logs/mcnn').mkdir(parents=True, exist_ok=True)
		from libs.models import MCNN
		model = MCNN(input_shape, learning_rate, epochs)
		if args.train == True:
			model.train()
		else:
			print("You didn't even try :(")
			
	if model_name == 'mcnnd':
		pathlib.Path('./logs/mcnnd').mkdir(parents=True, exist_ok=True)
		from libs.models import MCNNd
		model = MCNNd(input_shape, learning_rate, epochs)
		if args.train == True:
			model.train()
		else:
			print("You didn't even try :(")
			
	if model_name == 'tcnn':
		pathlib.Path('./logs/tcnn').mkdir(parents=True, exist_ok=True)
		from libs.models import TCNN, batch_data_generation
		model = TCNN(input_shape, learning_rate, epochs)
		if args.train == 'True':
			model.train('./data/df_tropical_cyclones.p')
		else:
			print("You didn't even try :(")
		
if __name__ == '__main__':
	main()

 
      	 	