import argparse
import warnings
import datetime
import os

def parse_args(args):
	""" Parse the arguments.
		"""
	parser = argparse.ArgumentParser(description='Training script for training MCNN,MCNNd, TCNN')


	parser.add_argument('--model', help='Which model will be used/trained', default="mcnn", type=str)
	parser.add_argument('--input_shape', help='Shape of input images', type= str, default='200,200,2')
	parser.add_argument('--learning_rate', help='Learning rate for training the model', type=float, default=0.00001)

	#parser.add_argument('--run-name',           dest="run_name",            help='name for the current run (directories will be created based on this name)', default='test_run')
	# parser.add_argument('--val-batch-size',     dest='val_batch_size',      help='Size of the batches for evaluation.', default=32, type=int)
	parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=300)
	#parser.add_argument('--wass-metric',        dest="wass_metric",         help="Flag for Wasserstein metric", action='store_true')
	# parser.add_argument('-–gpu',                dest="gpu",                 help="GPU id to use", default=0, type=int)
	#parser.add_argument('--num-workers',        dest="num_workers",         help="Number of dataset workers", default=1, type=int)
	#parser.add_argument('--snapshots-period',   dest="snapshots_period",    help="Save model snapshots every N epochs; -1 (default) for final models only", default=-1, type=int)
	#parser.add_argument('--latent-dim',         dest='latent_dim',          help='(real) embeddings dimensionality', type=int, default=32)
	#parser.add_argument('--cat-num',            dest='cat_num',             help='(int) embeddings dimensionality: number of categories', type=int, default=10)
	# parser.add_argument('--steps-per-epoch',        help='Number of steps per epoch.', type=int)
	# parser.add_argument('--val-steps',              help='Number of steps per validation run.', type=int, default=100)
	# parser.add_argument('--no-snapshots',           help='Disable saving snapshots.', dest='snapshots', action='store_false')
	#parser.add_argument('--debug',                  help='launch in DEBUG mode', dest='debug', action='store_true')

	parser.add_argument('--train', help='Train model or not', action='store_true', default=True)
	# parser.add_argument('--generator-multiplier',   help='generator batches per discriminator batches for the sake of training stability', dest='generator_multiplier', default=4, type=int)

	return parser.parse_args(args)

