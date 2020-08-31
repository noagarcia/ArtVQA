import numpy as np
import json
from sklearn.linear_model import LogisticRegression
import pickle
import argparse
import os
import neptune


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='processed', help='raw or processed')
    args, unknown = parser.parse_known_args()
    return args


def train(args):
	train_features = np.load("Cache Data/%s/train.npy" % args.data_type)
	labels = []
	with open("../Dataset/%s/train.json" % args.data_type) as f:
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
	test_features = np.load("Cache Data/%s/%s.npy" % (args.data_type, split))
	labels = []
	with open("../Dataset/%s/%s.json" % (args.data_type, split)) as f:
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
	neptune.log_metric('%s score' % split, acc)


def predict(classifier, split):
	test_features = np.load("Cache Data/%s/%s.npy" % (args.data_type, split))
	labels = classifier.predict(test_features)
	with open("../Dataset/%s/%s.json" % (args.data_type, split)) as f:
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
	with open("Results/%s_%s_need_kg.json" % (args.data_type, split), "w") as f:
		json.dump(need_external_knowledge, f, indent = 4)
	with open("Results/%s_%s_not_need_kg.json" % (args.data_type, split), "w") as f:
		json.dump(not_need_external_knowledge, f, indent = 4)

	neptune.log_artifact('Results/%s_%s_need_kg.json' % (args.data_type, split))
	neptune.log_artifact('Results/%s_%s_not_need_kg.json' % (args.data_type, split))


def gtlabel(split):
	with open("../Dataset/%s/%s.json" % (args.data_type, split)) as f:
		dataset = json.load(f)

	need_external_knowledge = []
	not_need_external_knowledge = []
	for i, item in enumerate(dataset):
		if item["need_external_knowledge"]:
			need_external_knowledge.append(item)
		else:
			not_need_external_knowledge.append(item)
	with open("Results/%s_%s_gt_kg.json" % (args.data_type, split), "w") as f:
		json.dump(need_external_knowledge, f, indent = 4)
	with open("Results/%s_%s_gt_not_kg.json" % (args.data_type, split), "w") as f:
		json.dump(not_need_external_knowledge, f, indent = 4)


if __name__ == "__main__":

	args = parse_args()
	assert args.data_type in ['raw', 'processed']

	# neptune.init('artQA/artQA-IJICAI-submission')
	# neptune.create_experiment(name="External Knowledge Classifier", upload_stdout=True, params=args.__dict__)
	# neptune.append_tags('kg class')
	# neptune.append_tags('%s data' % args.data_type)

	if not os.path.exists("Results"):
		os.makedirs("Results")

	# # Train
	filename = 'model_save_%s.pkl' % args.data_type
	# classifier = train(args)
	# pickle.dump(classifier, open(filename, 'wb'))

	# # Evaluation
	print(filename)
	classifier = pickle.load(open(filename, 'rb'))
	validation(classifier, split='test', print_confusin_matrix=True)
	# predict(classifier, split='test')
	# predict(classifier, split='val')
	# predict(classifier, split='train')
	# gtlabel(split='test')
	# neptune.stop()
