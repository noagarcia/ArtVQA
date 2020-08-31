import numpy as np
from bert_serving.client import BertClient
import json
import os

bc = BertClient()


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

	cache_dir = "Cache/"
	aqua_dir = "../../AQUA/"
	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)

	if os.path.exists(os.path.join(cache_dir, "train.npy")):
		train_features = np.load(os.path.join(cache_dir, "train.npy"))
		print(train_features.shape)
	else:
		train_features = load_dataset(os.path.join(aqua_dir, "train.json"), None)
		np.save(os.path.join(cache_dir, "train.npy"), train_features)
	print("Training Set Finished")

	if os.path.exists(os.path.join(cache_dir, "val.npy")):
		val_features = np.load(os.path.join(cache_dir, "val.npy"))
		print(val_features.shape)
	else:
		val_features = load_dataset(os.path.join(aqua_dir, "val.json"), None)
		np.save(os.path.join(cache_dir, "val.npy"), val_features)
	print("Validation Set Finished")

	if os.path.exists(os.path.join(cache_dir, "test.npy")):
		test_features = np.load(os.path.join(cache_dir, "test.npy"))
		print(test_features.shape)
	else:
		test_features = load_dataset(os.path.join(aqua_dir, "test.json"), None)
		np.save(os.path.join(cache_dir, "test.npy"), test_features)
	print("Test Set Finished")


	
