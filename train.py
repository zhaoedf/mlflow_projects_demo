import argparse
import mlflow
import numpy as np


def get_args():
	parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-r', '--epochs', metavar='E', type=float, default=0.1,
						help='Number of epochs', dest='epochs')
	parser.add_argument('-f', '--batch-size',
						help='Batch size', dest='batchsize')
						
	return parser.parse_args()
						
if __name__ == '__main__':
	args = get_args() 
	print(args.epochs)
	print(args.batchsize)
	client = MlflowClient(tracking_uri='http://localhost:10500')

	data = np.random.random(100)

	for idx,item in enumerate(data):
	    client.log_metric(run.info.run_id,'data',item,step=idx)
