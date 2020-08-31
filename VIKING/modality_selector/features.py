import numpy as np
from bert_serving.client import BertClient
import json
import os
import argparse

bc = BertClient()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='processed', help='raw or processed')
    args, unknown = parser.parse_known_args()
    return args

def load_dataset(path, size):
	with open(path) as f:
		dataset = json.load(f)

	with open("image_features.json") as f:
		image_features_dict = json.load(f)

	if size is None:
		size = len(dataset)
	questions = []
	image_features = []
	for item in dataset[:size]:
		questions.append(item["question"])
		image = item["image"]
		image_features.append(np.array(image_features_dict[image]))

	image_features = np.array(image_features)
	question_features = bc.encode(questions)
	features = np.concatenate((image_features, question_features), axis = 1)
	return features

if __name__ == "__main__":

	args = parse_args()
	assert args.data_type in ['raw', 'processed']

	if os.path.exists("Cache Data/%s/train.npy" % args.data_type):
		train_features = np.load("Cache Data/%s/train.npy" % args.data_type)
		print(train_features.shape)
	else:
		train_features = load_dataset("../Dataset/%s/train.json" % args.data_type, None)
		np.save("Cache Data/%s/train.npy" % args.data_type, train_features)
	print("Training Set Finished")

	if os.path.exists("Cache Data/%s/val.npy" % args.data_type):
		val_features = np.load("Cache Data/%s/val.npy" % args.data_type)
		print(val_features.shape)
	else:
		val_features = load_dataset("../Dataset/%s/val.json" % args.data_type, None)
		np.save("Cache Data/%s/val.npy" % args.data_type, val_features)
	print("Validation Set Finished")

	if os.path.exists("Cache Data/%s/test.npy" % args.data_type):
		test_features = np.load("Cache Data/%s/test.npy" % args.data_type)
		print(test_features.shape)
	else:
		test_features = load_dataset("../Dataset/%s/test.json" % args.data_type, None)
		np.save("Cache Data/%s/test.npy" % args.data_type, test_features)
	print("Test Set Finished")


	
