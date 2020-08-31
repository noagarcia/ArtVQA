import numpy as np
import json
from sklearn.linear_model import LogisticRegression
import pickle
import os

def train():
	train_features = np.load("Cache/train.npy")
	labels = []
	with open("../../AQUA/train.json") as f:
		dataset = json.load(f)
	for item in dataset:
		if item["need_external_knowledge"]:
			labels.append(1.0)
		else:
			labels.append(0.0)
	labels = np.array(labels)
	classifier = LogisticRegression().fit(train_features, labels)
	return classifier


def validation(classifier, split, print_confusin_matrix = False):
	test_features = np.load("Cache/%s.npy" % (split))
	labels = []
	with open("../../AQUA/%s.json" % (split)) as f:
		dataset = json.load(f)
	for item in dataset:
		if item["need_external_knowledge"]:
			labels.append(1.0)
		else:
			labels.append(0.0)
	labels = np.array(labels)
	acc = classifier.score(test_features, labels)
	if print_confusin_matrix:
		predict = classifier.predict(test_features)
		from sklearn.metrics import confusion_matrix
		print(confusion_matrix(labels, predict))
	print("Accuracy {}: {}".format(split, acc))


def predict(classifier, split):
	test_features = np.load("Cache/%s.npy" % (split))
	labels = classifier.predict(test_features)
	with open("../../AQUA/%s.json" % (split)) as f:
		dataset = json.load(f)

	need_external_knowledge = []
	not_need_external_knowledge = []
	for i, item in enumerate(dataset):
		if labels[i] == 0.0:
			item["predict_external_knowledge_predict"] = False
			not_need_external_knowledge.append(item)
		else:
			item["predict_external_knowledge_predict"] = True
			need_external_knowledge.append(item)
	with open("Results/%s_need_kg.json" % (split), "w") as f:
		json.dump(need_external_knowledge, f, indent = 4)
	with open("Results/%s_not_need_kg.json" % (split), "w") as f:
		json.dump(not_need_external_knowledge, f, indent = 4)


if __name__ == "__main__":

	if not os.path.exists("Results"):
		os.makedirs("Results")

	# Train
	filename = 'model_save.pkl'
	classifier = train()
	pickle.dump(classifier, open(filename, 'wb'))

	# Evaluation
	classifier = pickle.load(open(filename, 'rb'))
	validation(classifier, split='test', print_confusin_matrix=True)
	predict(classifier, split='test')
	predict(classifier, split='val')
	predict(classifier, split='train')
