import sys
import os
import traceback
import pathlib


def DoesPathExistAndIsDirectory(pathStr):
	if os.path.exists(pathStr) and os.path.isdir(pathStr):
		return True
	else:
		return False

def EnsureDirectoryExists(pathStr):
	if not DoesPathExistAndIsDirectory(pathStr):
		try:
			pathlib.Path(pathStr).mkdir(parents=True, exist_ok=True)
		except Exception as ex:
			err_fname = './errors.log'
			exc_type, exc_value, exc_traceback = sys.exc_info()
			with open(err_fname, 'a') as errf:
				traceback.print_tb(exc_traceback, limit=None, file=errf)
				traceback.print_exception(exc_type, exc_value, exc_traceback, limit=None, file=errf)
			print(str(ex))
			print('the directory you are trying to place a file to doesn\'t exist and cannot be created:\n%s' % pathStr)
			raise FileNotFoundError('the directory you are trying to place a file to doesn\'t exist and cannot be created:')