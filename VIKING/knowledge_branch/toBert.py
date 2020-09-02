import pandas as pd 
import csv
import random
import json

#construct the dictionary (key: imgId, val: comment)
img2comment = {}
train_file = open("./comments/semart_train.csv", encoding='mac_roman')
csv_reader = csv.reader(train_file, delimiter='\t')
line_count = 0
for row in csv_reader:
    if line_count == 0:
        print(row[0], row[1])
        line_count += 1
    else:
        img2comment[row[0]] = row[1]
        line_count += 1
print(f'Processed {line_count} lines.')
val_file = open("./comments/semart_val.csv", encoding='mac_roman')
csv_reader = csv.reader(val_file, delimiter='\t')
line_count = 0
for row in csv_reader:
    if line_count == 0:
        print(row[0], row[1])
        line_count += 1
    else:
        img2comment[row[0]] = row[1]
        line_count += 1
print(f'Processed {line_count} lines.')
test_file = open("./comments/semart_test.csv", encoding='mac_roman')
csv_reader = csv.reader(test_file, delimiter='\t')
line_count = 0
for row in csv_reader:
    if line_count == 0:
        print(row[0], row[1])
        line_count += 1
    else:
        img2comment[row[0]] = row[1]
        line_count += 1
print(f'Processed {line_count} lines.')

# read in json file
col = ['QUESTION', 'COMMENT', 'LABEL']
train_data = []
val_data = []
test_data = []

with open('./Data/comment_prediction_train.json') as json_file:
    content = json.load(json_file)
    for example in content:
        question = example['question']
        positive = example['image']
        negative = example['comments_prediction_top10'][0] if example['comments_prediction_top10'][0] != positive else example['comments_prediction_top10'][1]
        train_data.append([question, img2comment[positive], 1])
        train_data.append([question, img2comment[negative], 0])

with open('./Data/comment_prediction_val.json') as json_file:
    content = json.load(json_file)
    for example in content:
        question = example['question']
        positive = example['image']
        negative = example['comments_prediction_top10'][0] if example['comments_prediction_top10'][0] != positive else example['comments_prediction_top10'][1]
        val_data.append([question, img2comment[positive], 1])
        val_data.append([question, img2comment[negative], 0])

with open('./Data/comment_prediction_test.json') as json_file:
    content = json.load(json_file)
    for example in content:
        question = example['question']
        for img in example['comments_prediction_top10']:
            if img == example['image']:
                test_data.append([question, img2comment[img], 1])
            else:
                test_data.append([question, img2comment[img], 0])


train_df = pd.DataFrame(data=train_data, columns=col)
train_df.to_csv('./Cache/qaFromComments_train.csv', index=False)
val_df = pd.DataFrame(data=val_data, columns=col)
val_df.to_csv('./Cache/qaFromComments_val.csv', index=False)
test_df = pd.DataFrame(data=test_data, columns=col)
test_df.to_csv('./Cache/qaFromComments_test.csv', index=False)
